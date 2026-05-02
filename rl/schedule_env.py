"""
Single-day doctor scheduling MDP: assign each patient (GPT 3-vector) to a doctor
and start time slot without illegal overlaps.

Illegal overlaps:
- Same doctor cannot cover two patients with intersecting [start, end) intervals.
- Optional per routing_hint concurrent cap (shared departmental capacity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.mapping import ROUTING_HINT_BUCKETS, routing_hint_label


# Default service length in slots (30 min per slot) by routing_hint_id index.
DEFAULT_DURATION_SLOTS: Tuple[int, ...] = (2, 4, 2, 3, 2, 1, 2, 2)

# Max concurrent patients overlapping in time per routing_hint_id (departmental pressure).
DEFAULT_DEPT_CONCURRENT_CAP: Tuple[int, ...] = (10, 5, 14, 6, 8, 20, 16, 12)


@dataclass
class DoctorSpec:
    id: int
    label: str
    hints: Tuple[int, ...]  # eligible routing_hint_id values


def default_doctors() -> List[DoctorSpec]:
    """Small synthetic pool: overlapping skills, multiple ED physicians."""
    return [
        DoctorSpec(0, "Dr. Ellis (ED)", (0, 7)),
        DoctorSpec(1, "Dr. Park (ED)", (0, 5, 7)),
        DoctorSpec(2, "Dr. Nguyen (Hospitalist)", (2, 4, 6, 7)),
        DoctorSpec(3, "Dr. Okonkwo (ICU)", (1, 2)),
        DoctorSpec(4, "Dr. Patel (ICU)", (1,)),
        DoctorSpec(5, "Dr. Rivera (OB)", (3, 5, 7)),
        DoctorSpec(6, "Dr. Santos (Outpatient)", (5, 6, 7)),
        DoctorSpec(7, "Dr. Kim (Flex)", (0, 2, 7)),
    ]


def _slot_from_arrival_fraction(f: float, num_slots: int) -> int:
    s = int(float(f) * num_slots)
    return max(0, min(num_slots - 1, s))


def load_patients_from_rows(
    rows: Sequence[Dict[str, Any]],
    num_slots: int,
    duration_by_hint: Sequence[int] | None = None,
) -> List[Dict[str, Any]]:
    """Sort by arrival, attach integer arrival_slot and duration_slots."""
    dmap = duration_by_hint or DEFAULT_DURATION_SLOTS
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        hint = int(r["routing_hint_id"])
        hint = max(0, min(len(ROUTING_HINT_BUCKETS) - 1, hint))
        dur = dmap[hint] if hint < len(dmap) else 2
        out.append(
            {
                "patient_event_id": int(r.get("patient_event_id", i)),
                "token_diagnosis_id": int(r["token_diagnosis_id"]),
                "arrival_fraction_day": float(r["arrival_fraction_day"]),
                "routing_hint_id": hint,
                "arrival_slot": _slot_from_arrival_fraction(float(r["arrival_fraction_day"]), num_slots),
                "duration_slots": int(dur),
            }
        )
    out.sort(key=lambda x: (x["arrival_slot"], x["patient_event_id"]))
    return out


class DoctorDayScheduleEnv(gym.Env):
    """
    One decision per patient (in arrival order): choose doctor d and start slot s.

    Action is a single integer in [0, n_doctors * n_slots) encoding s + d * n_slots.
    Illegal masked actions receive logit mask -inf in the policy; env still validates.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        patients: List[Dict[str, Any]],
        doctors: List[DoctorSpec] | None = None,
        num_slots: int = 48,
        duration_by_hint: Sequence[int] | None = None,
        dept_concurrent_cap: Sequence[int] | None = None,
        wait_penalty: float = 1.0,
        mismatch_penalty: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.doctors = doctors or default_doctors()
        self.n_docs = len(self.doctors)
        self.duration_by_hint = tuple(duration_by_hint or DEFAULT_DURATION_SLOTS)
        self.dept_cap = tuple(dept_concurrent_cap or DEFAULT_DEPT_CONCURRENT_CAP)
        self.wait_penalty = float(wait_penalty)
        self.mismatch_penalty = float(mismatch_penalty)

        self.patients = patients
        self.n_patients = len(patients)
        self.n_hints = len(ROUTING_HINT_BUCKETS)

        self.action_space = spaces.Discrete(self.n_docs * self.num_slots)

        # Obs: routing one-hot (n_hints) + arrival/T + duration/maxDur + diag/log + per-doc earliest_free/T
        obs_dim = self.n_hints + 2 + 1 + self.n_docs
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self._reset_internal()

    def _reset_internal(self) -> None:
        self._step_idx = 0
        # doctor d busy in [busy_start[d], busy_end[d]) as sorted disjoint intervals per doctor
        self._intervals: List[List[Tuple[int, int]]] = [[] for _ in range(self.n_docs)]
        # dept occupancy: for each hint, array of length T with count of patients covering that slot
        self._dept_load = np.zeros((self.n_hints, self.num_slots), dtype=np.int32)
        self._assignments: List[Dict[str, Any]] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._reset_internal()
        return self._obs(), {"legal_count": self._legal_mask().sum()}

    def _earliest_free(self, doc_idx: int) -> int:
        if not self._intervals[doc_idx]:
            return 0
        return max(end for _, end in self._intervals[doc_idx])

    def _obs(self) -> np.ndarray:
        if self._step_idx >= self.n_patients:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        p = self.patients[self._step_idx]
        h = int(p["routing_hint_id"])
        oh = np.zeros(self.n_hints, dtype=np.float32)
        oh[h] = 1.0
        arr = float(p["arrival_slot"]) / max(1, self.num_slots - 1)
        dur = float(p["duration_slots"]) / max(1, max(self.duration_by_hint))
        diag = min(1.0, np.log1p(float(p["token_diagnosis_id"])) / np.log1p(2048.0))
        frees = []
        for d in range(self.n_docs):
            frees.append(self._earliest_free(d) / max(1, self.num_slots - 1))
        return np.concatenate([oh, [arr, dur, diag], np.array(frees, dtype=np.float32)], axis=0)

    def _legal_mask(self) -> np.ndarray:
        """Boolean mask of shape (n_actions,) for flat actions."""
        mask = np.zeros(self.n_docs * self.num_slots, dtype=bool)
        if self._step_idx >= self.n_patients:
            return mask
        p = self.patients[self._step_idx]
        arrival = int(p["arrival_slot"])
        dur = int(p["duration_slots"])
        hint = int(p["routing_hint_id"])
        cap = int(self.dept_cap[hint]) if hint < len(self.dept_cap) else 8

        for d, doc in enumerate(self.doctors):
            if hint not in doc.hints:
                continue
            for start in range(self.num_slots):
                end = start + dur
                if end > self.num_slots:
                    break
                if start < arrival:
                    continue
                if not self._doctor_free(d, start, end):
                    continue
                if not self._dept_has_room(hint, start, end, cap):
                    continue
                mask[start + d * self.num_slots] = True
        return mask

    def _doctor_free(self, doc_idx: int, start: int, end: int) -> bool:
        for a, b in self._intervals[doc_idx]:
            if not (end <= a or start >= b):
                return False
        return True

    def _dept_has_room(self, hint: int, start: int, end: int, cap: int) -> bool:
        for t in range(start, end):
            if self._dept_load[hint, t] >= cap:
                return False
        return True

    def _add_interval(self, doc_idx: int, start: int, end: int) -> None:
        self._intervals[doc_idx].append((start, end))
        self._intervals[doc_idx].sort()

    def _bump_dept(self, hint: int, start: int, end: int, delta: int) -> None:
        self._dept_load[hint, start:end] += delta

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._step_idx >= self.n_patients:
            return self._obs(), 0.0, True, False, {"reason": "already_done"}

        p = self.patients[self._step_idx]
        arrival = int(p["arrival_slot"])
        dur = int(p["duration_slots"])
        hint = int(p["routing_hint_id"])
        mask = self._legal_mask()

        d = int(action) // self.num_slots
        start = int(action) % self.num_slots
        end = start + dur
        legal = 0 <= action < mask.size and mask[action]

        if not legal:
            # Penalize invalid moves without advancing (keeps MDP well-defined for masked policy).
            obs = self._obs()
            return obs, -25.0, False, False, {"invalid": True, "legal_count": int(mask.sum())}

        wait = max(0, start - arrival)
        reward = -self.wait_penalty * wait
        # Mild pressure to align with primary department line (already enforced by hints).
        cap = int(self.dept_cap[hint]) if hint < len(self.dept_cap) else 8
        load_term = float(self._dept_load[hint, start:end].max()) / max(1, cap)
        reward -= self.mismatch_penalty * load_term

        self._add_interval(d, start, end)
        self._bump_dept(hint, start, end, 1)
        self._assignments.append(
            {
                "patient_event_id": p["patient_event_id"],
                "token_diagnosis_id": p["token_diagnosis_id"],
                "arrival_fraction_day": p["arrival_fraction_day"],
                "routing_hint_id": hint,
                "routing_hint_name": routing_hint_label(hint),
                "doctor_id": self.doctors[d].id,
                "doctor_label": self.doctors[d].label,
                "department_assigned": ROUTING_HINT_BUCKETS[hint],
                "start_slot": start,
                "end_slot": end,
            }
        )
        self._step_idx += 1
        terminated = self._step_idx >= self.n_patients
        obs = self._obs()
        info = {
            "assignments": list(self._assignments),
            "legal_count": int(self._legal_mask().sum()) if not terminated else 0,
        }
        return obs, float(reward), terminated, False, info

    def get_action_mask(self) -> np.ndarray:
        return self._legal_mask().astype(np.float32)

    def assignments(self) -> List[Dict[str, Any]]:
        return list(self._assignments)
