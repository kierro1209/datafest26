#!/usr/bin/env python3
"""
Train a REINFORCE policy to assign doctors and start slots for one day of GPT vectors,
then export a schedule CSV (no overlapping assignments per doctor; department caps enforced).

Example:
  python rl/train_doctor_scheduler.py \\
    --patients rl/synthetic/gpt_planner_train.csv \\
    --max-patients 24 \\
    --date 2026-05-01 \\
    --episodes 4000 \\
    --out rl/output/schedule.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.optim as optim

from rl.schedule_env import DoctorDayScheduleEnv, default_doctors, load_patients_from_rows
from rl.schedule_policy import SchedulePolicyMLP


def load_patient_rows_csv(path: Path, max_patients: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i >= max_patients:
                break
            d = dict(row)
            raw_id = d.get("patient_event_id")
            try:
                d["patient_event_id"] = int(raw_id) if raw_id not in (None, "") else i
            except (TypeError, ValueError):
                d["patient_event_id"] = i
            rows.append(d)
    return rows


def slots_to_hhmm(start_slot: int, end_slot: int, slot_minutes: int) -> Tuple[str, str]:
    def fmt(slot: int) -> str:
        m = slot * slot_minutes
        h, mm = divmod(m, 60)
        return f"{h:02d}:{mm:02d}"

    return fmt(start_slot), fmt(end_slot)


def export_schedule_csv(
    path: Path,
    schedule_date: str,
    assignments: List[Dict[str, Any]],
    slot_minutes: int,
    episode_total_reward: Optional[float] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "schedule_date",
        "patient_event_id",
        "token_diagnosis_id",
        "arrival_fraction_day",
        "routing_hint_id",
        "routing_hint_name",
        "doctor_id",
        "doctor_label",
        "department_assigned",
        "resource_line",
        "start_slot",
        "end_slot",
        "start_time_hhmm",
        "end_time_hhmm",
        "slot_duration_minutes",
    ]
    if episode_total_reward is not None:
        fieldnames.append("episode_total_reward")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for a in assignments:
            st, et = slots_to_hhmm(int(a["start_slot"]), int(a["end_slot"]), slot_minutes)
            row = {
                "schedule_date": schedule_date,
                "patient_event_id": a["patient_event_id"],
                "token_diagnosis_id": a["token_diagnosis_id"],
                "arrival_fraction_day": a["arrival_fraction_day"],
                "routing_hint_id": a["routing_hint_id"],
                "routing_hint_name": a["routing_hint_name"],
                "doctor_id": a["doctor_id"],
                "doctor_label": a["doctor_label"],
                "department_assigned": a["department_assigned"],
                "resource_line": f"{a['department_assigned']}|doctor_{a['doctor_id']}",
                "start_slot": a["start_slot"],
                "end_slot": a["end_slot"],
                "start_time_hhmm": st,
                "end_time_hhmm": et,
                "slot_duration_minutes": slot_minutes,
            }
            if episode_total_reward is not None:
                row["episode_total_reward"] = f"{episode_total_reward:.6f}"
            w.writerow(row)


def greedy_schedule(
    env: DoctorDayScheduleEnv,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[float]]:
    """
    Minimum-wait feasible assignment in arrival order.

    Returns (assignments, episode_total_reward). Reward is the sum of env step
    rewards (same definition as RL). On infeasible partial schedule, returns
    (None, None).
    """
    env.reset()
    episode_total_reward = 0.0
    while env._step_idx < env.n_patients:
        mask = env.get_action_mask()
        if float(mask.sum()) < 0.5:
            return None, None
        p = env.patients[env._step_idx]
        arrival = int(p["arrival_slot"])
        best_wait = None
        best_a = None
        for a in np.where(mask > 0.5)[0]:
            start = int(a) % env.num_slots
            wait = max(0, start - arrival)
            if best_wait is None or wait < best_wait or (wait == best_wait and int(a) < best_a):
                best_wait = wait
                best_a = int(a)
        assert best_a is not None
        _, r, term, _, _ = env.step(best_a)
        episode_total_reward += float(r)
        if term:
            break
    return env.assignments(), episode_total_reward


def validate_schedule(assignments: List[Dict[str, Any]]) -> bool:
    """No overlapping intervals per doctor_id."""
    by_doc: Dict[int, List[Tuple[int, int]]] = {}
    for a in assignments:
        d = int(a["doctor_id"])
        by_doc.setdefault(d, []).append((int(a["start_slot"]), int(a["end_slot"])))
    for intervals in by_doc.values():
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                return False
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patients", type=Path, required=True, help="CSV with GPT planner columns")
    p.add_argument("--max-patients", type=int, default=24, help="First N rows after sort = one day")
    p.add_argument("--date", type=str, default="2026-05-01", help="schedule_date column in output")
    p.add_argument("--num-slots", type=int, default=48, help="Slots per day (e.g. 48 × 30min = 24h)")
    p.add_argument("--slot-minutes", type=int, default=30, help="Minutes per slot for CSV clock times")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", type=Path, default=_REPO_ROOT / "rl" / "output" / "schedule.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--greedy-only", action="store_true", help="Skip RL; export greedy feasible schedule")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rows = load_patient_rows_csv(args.patients, args.max_patients)
    if not rows:
        raise SystemExit("No patient rows loaded.")

    patients = load_patients_from_rows(rows, num_slots=args.num_slots)
    doctors = default_doctors()
    env = DoctorDayScheduleEnv(patients, doctors, num_slots=args.num_slots)

    # Feasibility check + greedy baseline score (same reward definition as RL)
    fe_env = DoctorDayScheduleEnv(patients, doctors, num_slots=args.num_slots)
    g, greedy_reward = greedy_schedule(fe_env)
    if g is None or len(g) != len(patients) or greedy_reward is None:
        raise SystemExit(
            "Greedy could not build a full schedule: relax caps, add doctors, or reduce patients."
        )

    if args.greedy_only:
        export_schedule_csv(
            args.out, args.date, g, args.slot_minutes, episode_total_reward=greedy_reward
        )
        print(
            f"Exported greedy schedule ({len(g)} rows) to {args.out.resolve()} "
            f"(episode_total_reward={greedy_reward:.4f})"
        )
        return

    obs0, _ = env.reset(seed=0)
    obs_dim = obs0.shape[0]
    n_actions = env.action_space.n
    policy = SchedulePolicyMLP(obs_dim, n_actions).to(device)
    opt = optim.Adam(policy.parameters(), lr=args.lr)

    best_reward = -1e18
    best_assign: Optional[List[Dict[str, Any]]] = None
    baseline = 0.0

    rng = np.random.default_rng(args.seed)
    for ep in range(args.episodes):
        seed = int(rng.integers(0, 2**31 - 1))
        env_ep = DoctorDayScheduleEnv(patients, doctors, num_slots=args.num_slots)
        obs, _ = env_ep.reset(seed=seed)
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        ok = True
        for _ in range(env_ep.n_patients):
            mask_np = env_ep.get_action_mask()
            if float(mask_np.sum()) < 0.5:
                ok = False
                break
            o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            m = torch.tensor(mask_np > 0.5, dtype=torch.bool, device=device).unsqueeze(0)
            dist = policy.action_distribution(o, m)
            a = dist.sample()
            log_probs.append(dist.log_prob(a))
            obs, r, term, _, _ = env_ep.step(int(a.item()))
            rewards.append(float(r))
            if term:
                break
        if not ok or not log_probs:
            continue
        total_r = sum(rewards)
        assigns = env_ep.assignments()
        if len(assigns) == len(patients) and total_r > best_reward:
            best_reward = total_r
            best_assign = list(assigns)

        G = total_r
        baseline = 0.95 * baseline + 0.05 * G
        lp = torch.stack(log_probs).sum()
        loss = -lp * (G - baseline)
        opt.zero_grad()
        loss.backward()
        opt.step()

    if best_assign is None:
        best_assign = g
        export_reward = greedy_reward
        score_note = (
            f"greedy baseline only (RL did not improve); "
            f"episode_total_reward={greedy_reward:.4f}"
        )
    else:
        export_reward = best_reward
        score_note = (
            f"best RL episode_total_reward={best_reward:.4f} "
            f"(greedy baseline={greedy_reward:.4f})"
        )
    if not validate_schedule(best_assign):
        raise RuntimeError("Internal error: exported schedule has doctor overlap.")
    export_schedule_csv(
        args.out, args.date, best_assign, args.slot_minutes, episode_total_reward=export_reward
    )
    print(f"Exported schedule ({len(best_assign)} rows) to {args.out.resolve()} ({score_note})")


if __name__ == "__main__":
    main()
