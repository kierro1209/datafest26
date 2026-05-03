#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment, misc]

log = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[BOS]": 1,
    "[EOS]": 2,
    "[UNK]": 3,
}


@dataclass
class PreparedData:
    sequences: list[dict[str, Any]]
    type_to_id: dict[str, int]
    event_description_to_id: dict[str, int]
    group_code_to_id: dict[str, int]
    diagnosis_value_to_id: dict[str, int]
    gap_to_id: dict[str, int]
    setting_to_id: dict[str, int]
    dept_type_to_id: dict[str, int]
    dept_specialty_to_id: dict[str, int]
    facility_size_to_id: dict[str, int]
    region_to_id: dict[str, int]
    patient_context_to_id: dict[str, Any]
    sdoh_status_to_id: dict[str, int]
    sdoh_fields: list[str]
    patient_context_fields: list[str]
    patient_numeric_fields: list[str]
    metadata: dict[str, Any]
    # Optional per-timestep streams beyond core WHAT/WHEN/WHERE (e.g. census / market bins).
    external_stream_bases: list[str] = field(default_factory=list)
    external_to_id: dict[str, dict[str, int]] = field(default_factory=dict)
    # Per-timestep float vector (e.g. ZHVI, ADI, FCC, NIBRS) from final_token_format companion PT.
    external_per_token_dim: int = 0
    external_feature_key: str = "external_feature_tensor"


# Per-target-timestep roles for temporal split (aligned with target_* tensors).
SPLIT_IGNORE = 0
SPLIT_TRAIN = 1
SPLIT_VAL = 2
SPLIT_TEST = 3

HEAD_TO_TARGET: dict[str, str] = {
    "type": "type",
    "event_description": "event_description",
    "group_code": "group_code",
    "diagnosis_value": "diagnosis_value",
    "gap": "gap",
    "setting": "setting",
    "dept_type": "dept_type",
    "dept_specialty": "dept_specialty",
    "facility_size": "facility_size",
    "region": "region",
}

CANONICAL_STREAM_ID_KEYS = frozenset(
    {
        "type_ids",
        "event_description_ids",
        "group_code_ids",
        "diagnosis_value_ids",
        "gap_ids",
        "setting_ids",
        "dept_type_ids",
        "dept_specialty_ids",
        "facility_size_ids",
        "region_ids",
    }
)


def merge_head_to_target(external_stream_bases: list[str]) -> dict[str, str]:
    m = dict(HEAD_TO_TARGET)
    for base in external_stream_bases:
        m[base] = base
    return m


def discover_external_stream_bases(sample: dict[str, Any], sdoh_fields: list[str]) -> list[str]:
    """Keys like `<base>_ids` with same length as `type_ids`, excluding core and SDOH streams."""
    type_ids = sample.get("type_ids")
    if not isinstance(type_ids, list) or len(type_ids) < 2:
        return []
    seq_len = len(type_ids)
    sdoh_set = set(sdoh_fields)
    bases: list[str] = []
    for key, val in sample.items():
        if key in CANONICAL_STREAM_ID_KEYS or key in sdoh_set:
            continue
        if not key.endswith("_ids"):
            continue
        if not isinstance(val, list) or len(val) != seq_len:
            continue
        base = key[: -len("_ids")]
        if base:
            bases.append(base)
    return sorted(bases)


def temporal_split_ends(seq_len: int, train_f: float, valid_f: float, test_f: float) -> tuple[int, int]:
    """Exclusive boundaries (end_train, end_val) on target indices [0, seq_len).

    Indices are **event positions** along the (possibly `--max-seq-len`-truncated) timeline.
    Train targets index i in [0, end_train), val in [end_train, end_val), test in [end_val, seq_len).
    """
    if seq_len < 2:
        return 0, 0
    s = train_f + valid_f + test_f
    if s <= 0:
        train_f, valid_f, test_f = 0.7, 0.15, 0.15
        s = 1.0
    train_f, valid_f, test_f = train_f / s, valid_f / s, test_f / s

    n_train = max(1, int(round(seq_len * train_f)))
    n_val = int(round(seq_len * valid_f))
    n_test = seq_len - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_val = max(0, seq_len - n_train - n_test)
    while n_train + n_val + n_test > seq_len and n_val > 0:
        n_val -= 1
        n_test = seq_len - n_train - n_val
    while n_train + n_val + n_test > seq_len and n_train > 1:
        n_train -= 1
        n_test = seq_len - n_train - n_val
    if n_train >= seq_len:
        n_train = seq_len - 1
    end_train = n_train
    end_val = n_train + n_val
    if end_val > seq_len:
        end_val = seq_len
    if end_train >= end_val:
        end_val = min(seq_len, end_train + 1)
    return end_train, end_val


class PatientSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[dict[str, Any]],
        max_seq_len: int,
        sdoh_fields: list[str],
        patient_context_fields: list[str],
        patient_numeric_fields: list[str],
        temporal_split: bool = False,
        train_fraction: float = 0.7,
        valid_fraction: float = 0.15,
        test_fraction: float = 0.15,
        external_stream_bases: list[str] | None = None,
        external_per_token_dim: int = 0,
        external_feature_key: str = "external_feature_tensor",
    ) -> None:
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.sdoh_fields = sdoh_fields
        self.patient_context_fields = patient_context_fields
        self.patient_numeric_fields = patient_numeric_fields
        self.temporal_split = temporal_split
        self.train_fraction = train_fraction
        self.valid_fraction = valid_fraction
        self.test_fraction = test_fraction
        self.external_stream_bases = list(external_stream_bases or [])
        self.external_per_token_dim = int(external_per_token_dim)
        self.external_feature_key = external_feature_key

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        type_ids = seq["type_ids"][: self.max_seq_len]
        event_description_ids = seq["event_description_ids"][: self.max_seq_len]
        group_code_ids = seq["group_code_ids"][: self.max_seq_len]
        diagnosis_value_ids = seq["diagnosis_value_ids"][: self.max_seq_len]
        gap_ids = seq["gap_ids"][: self.max_seq_len]
        setting_ids = seq["setting_ids"][: self.max_seq_len]
        dept_ids = seq["dept_type_ids"][: self.max_seq_len]
        dept_specialty_ids = seq["dept_specialty_ids"][: self.max_seq_len]
        size_ids = seq["facility_size_ids"][: self.max_seq_len]
        region_ids = seq["region_ids"][: self.max_seq_len]

        seq_len = len(type_ids)
        attention_mask = [1] * seq_len

        input_type_ids = [SPECIAL_TOKENS["[BOS]"]] + type_ids[:-1]
        input_event_description_ids = [SPECIAL_TOKENS["[BOS]"]] + event_description_ids[:-1]
        input_group_code_ids = [SPECIAL_TOKENS["[BOS]"]] + group_code_ids[:-1]
        input_diagnosis_value_ids = [SPECIAL_TOKENS["[BOS]"]] + diagnosis_value_ids[:-1]
        input_gap_ids = [SPECIAL_TOKENS["[BOS]"]] + gap_ids[:-1]
        input_setting_ids = [SPECIAL_TOKENS["[BOS]"]] + setting_ids[:-1]
        input_dept_type_ids = [SPECIAL_TOKENS["[BOS]"]] + dept_ids[:-1]
        input_dept_specialty_ids = [SPECIAL_TOKENS["[BOS]"]] + dept_specialty_ids[:-1]
        input_facility_size_ids = [SPECIAL_TOKENS["[BOS]"]] + size_ids[:-1]
        input_region_ids = [SPECIAL_TOKENS["[BOS]"]] + region_ids[:-1]

        sdoh_inputs: dict[str, list[int]] = {}
        for field in self.sdoh_fields:
            values = list(seq.get(field, []))[: self.max_seq_len]
            sdoh_inputs[field] = [SPECIAL_TOKENS["[BOS]"]] + values[:-1]

        external_inputs: dict[str, list[int]] = {}
        external_targets: dict[str, list[int]] = {}
        for base in self.external_stream_bases:
            col = f"{base}_ids"
            vals = list(seq.get(col, []))[: self.max_seq_len]
            external_inputs[base] = [SPECIAL_TOKENS["[BOS]"]] + vals[:-1]
            external_targets[base] = vals

        patient_context_ids = seq.get("patient_context_ids", {}) or {}
        patient_context_values = seq.get("patient_context_values", {}) or {}
        static_context_ids = [int(patient_context_ids.get(field, 0)) + len(SPECIAL_TOKENS) for field in self.patient_context_fields]
        static_numeric_values = []
        for field in self.patient_numeric_fields:
            value = patient_context_values.get(field, 0.0)
            try:
                static_numeric_values.append(float(value))
            except (TypeError, ValueError):
                static_numeric_values.append(0.0)

        pad_n = self.max_seq_len - seq_len
        if pad_n > 0:
            input_type_ids += [SPECIAL_TOKENS["[PAD]"]] * pad_n
            input_event_description_ids += [SPECIAL_TOKENS["[PAD]"]] * pad_n
            input_group_code_ids += [SPECIAL_TOKENS["[PAD]"]] * pad_n
            input_diagnosis_value_ids += [SPECIAL_TOKENS["[PAD]"]] * pad_n
            input_gap_ids += [SPECIAL_TOKENS["[PAD]"]] * pad_n
            input_setting_ids += [SPECIAL_TOKENS["[PAD]"]] * pad_n
            input_dept_type_ids += [SPECIAL_TOKENS["[PAD]"]] * pad_n
            input_dept_specialty_ids += [SPECIAL_TOKENS["[PAD]"]] * pad_n
            input_facility_size_ids += [SPECIAL_TOKENS["[PAD]"]] * pad_n
            input_region_ids += [SPECIAL_TOKENS["[PAD]"]] * pad_n
            for field in self.sdoh_fields:
                sdoh_inputs[field] += [SPECIAL_TOKENS["[PAD]"]] * pad_n
            for base in self.external_stream_bases:
                external_inputs[base] += [SPECIAL_TOKENS["[PAD]"]] * pad_n
                external_targets[base] += [-100] * pad_n
            type_ids += [-100] * pad_n
            event_description_ids += [-100] * pad_n
            group_code_ids += [-100] * pad_n
            diagnosis_value_ids += [-100] * pad_n
            gap_ids += [-100] * pad_n
            setting_ids += [-100] * pad_n
            dept_ids += [-100] * pad_n
            dept_specialty_ids += [-100] * pad_n
            size_ids += [-100] * pad_n
            region_ids += [-100] * pad_n
            attention_mask += [0] * pad_n

        split_role = [SPLIT_IGNORE] * self.max_seq_len
        if self.temporal_split:
            end_train, end_val = temporal_split_ends(seq_len, self.train_fraction, self.valid_fraction, self.test_fraction)
            for i in range(seq_len):
                if i < end_train:
                    split_role[i] = SPLIT_TRAIN
                elif i < end_val:
                    split_role[i] = SPLIT_VAL
                else:
                    split_role[i] = SPLIT_TEST

        ext_rows: list[list[float]] | None = None
        if self.external_per_token_dim > 0:
            ext_rows = _external_rows_to_matrix(seq.get(self.external_feature_key), seq_len, self.external_per_token_dim)
            if pad_n > 0:
                ext_rows += [[0.0] * self.external_per_token_dim for _ in range(pad_n)]

        batch = {
            "input_type_ids": torch.tensor(input_type_ids, dtype=torch.long),
            "input_event_description_ids": torch.tensor(input_event_description_ids, dtype=torch.long),
            "input_group_code_ids": torch.tensor(input_group_code_ids, dtype=torch.long),
            "input_diagnosis_value_ids": torch.tensor(input_diagnosis_value_ids, dtype=torch.long),
            "input_gap_ids": torch.tensor(input_gap_ids, dtype=torch.long),
            "input_setting_ids": torch.tensor(input_setting_ids, dtype=torch.long),
            "input_dept_type_ids": torch.tensor(input_dept_type_ids, dtype=torch.long),
            "input_dept_specialty_ids": torch.tensor(input_dept_specialty_ids, dtype=torch.long),
            "input_facility_size_ids": torch.tensor(input_facility_size_ids, dtype=torch.long),
            "input_region_ids": torch.tensor(input_region_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "static_context_ids": torch.tensor(static_context_ids, dtype=torch.long),
            "static_numeric_values": torch.tensor(static_numeric_values, dtype=torch.float32),
            "target_type": torch.tensor(type_ids, dtype=torch.long),
            "target_event_description": torch.tensor(event_description_ids, dtype=torch.long),
            "target_group_code": torch.tensor(group_code_ids, dtype=torch.long),
            "target_diagnosis_value": torch.tensor(diagnosis_value_ids, dtype=torch.long),
            "target_gap": torch.tensor(gap_ids, dtype=torch.long),
            "target_setting": torch.tensor(setting_ids, dtype=torch.long),
            "target_dept_type": torch.tensor(dept_ids, dtype=torch.long),
            "target_dept_specialty": torch.tensor(dept_specialty_ids, dtype=torch.long),
            "target_facility_size": torch.tensor(size_ids, dtype=torch.long),
            "target_region": torch.tensor(region_ids, dtype=torch.long),
        }
        if self.temporal_split:
            batch["split_role"] = torch.tensor(split_role, dtype=torch.long)
        for field in self.sdoh_fields:
            batch[f"input_{field}"] = torch.tensor(sdoh_inputs[field], dtype=torch.long)
        for base in self.external_stream_bases:
            batch[f"input_{base}_ids"] = torch.tensor(external_inputs[base], dtype=torch.long)
            batch[f"target_{base}"] = torch.tensor(external_targets[base], dtype=torch.long)
        if ext_rows is not None:
            batch[self.external_feature_key] = torch.tensor(ext_rows, dtype=torch.float32)
        return batch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PatientEventSequenceModel(nn.Module):
    def __init__(
        self,
        type_size: int,
        event_description_size: int,
        group_code_size: int,
        diagnosis_value_size: int,
        gap_size: int,
        setting_size: int,
        dept_type_size: int,
        dept_specialty_size: int,
        facility_size_size: int,
        region_size: int,
        sdoh_size: int,
        n_sdoh_streams: int,
        patient_context_vocab_sizes: list[int],
        patient_numeric_dim: int,
        d_model: int,
        max_seq_len: int,
        backbone: str,
        n_heads: int,
        n_layers: int,
        dropout: float,
        pad_token_id: int,
        external_stream_bases: list[str] | None = None,
        external_vocab_sizes: dict[str, int] | None = None,
        external_per_token_dim: int = 0,
        external_feature_key: str = "external_feature_tensor",
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.pad_token_id = pad_token_id
        self.external_stream_bases = list(external_stream_bases or [])
        ev = external_vocab_sizes or {}
        self.external_per_token_dim = int(external_per_token_dim)
        self.external_feature_batch_key = external_feature_key
        self.type_embedding = nn.Embedding(type_size, d_model, padding_idx=pad_token_id)
        self.event_description_embedding = nn.Embedding(event_description_size, d_model, padding_idx=pad_token_id)
        self.group_code_embedding = nn.Embedding(group_code_size, d_model, padding_idx=pad_token_id)
        self.diagnosis_value_embedding = nn.Embedding(diagnosis_value_size, d_model, padding_idx=pad_token_id)
        self.gap_embedding = nn.Embedding(gap_size, d_model, padding_idx=pad_token_id)
        self.setting_embedding = nn.Embedding(setting_size, d_model, padding_idx=pad_token_id)
        self.dept_type_embedding = nn.Embedding(dept_type_size, d_model, padding_idx=pad_token_id)
        self.dept_specialty_embedding = nn.Embedding(dept_specialty_size, d_model, padding_idx=pad_token_id)
        self.facility_size_embedding = nn.Embedding(facility_size_size, d_model, padding_idx=pad_token_id)
        self.region_embedding = nn.Embedding(region_size, d_model, padding_idx=pad_token_id)
        self.sdoh_embeddings = nn.ModuleList(
            nn.Embedding(sdoh_size, d_model, padding_idx=pad_token_id) for _ in range(n_sdoh_streams)
        )
        self.patient_context_embeddings = nn.ModuleList(
            nn.Embedding(vocab_size, d_model, padding_idx=0) for vocab_size in patient_context_vocab_sizes
        )
        self.patient_numeric_projection = nn.Linear(patient_numeric_dim, d_model) if patient_numeric_dim > 0 else None
        self.external_per_token_proj = (
            nn.Linear(self.external_per_token_dim, d_model) if self.external_per_token_dim > 0 else None
        )
        self.positional = PositionalEncoding(d_model=d_model, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

        if backbone == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        elif backbone == "gru":
            self.backbone = nn.GRU(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            self.backbone = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0.0,
                batch_first=True,
            )

        self.type_head = nn.Linear(d_model, type_size)
        self.event_description_head = nn.Linear(d_model, event_description_size)
        self.group_code_head = nn.Linear(d_model, group_code_size)
        self.diagnosis_value_head = nn.Linear(d_model, diagnosis_value_size)
        self.gap_head = nn.Linear(d_model, gap_size)
        self.setting_head = nn.Linear(d_model, setting_size)
        self.dept_type_head = nn.Linear(d_model, dept_type_size)
        self.dept_specialty_head = nn.Linear(d_model, dept_specialty_size)
        self.facility_size_head = nn.Linear(d_model, facility_size_size)
        self.region_head = nn.Linear(d_model, region_size)
        self.external_embeddings = nn.ModuleDict()
        self.external_heads = nn.ModuleDict()
        for base in self.external_stream_bases:
            sz = ev[base]
            self.external_embeddings[base] = nn.Embedding(sz, d_model, padding_idx=pad_token_id)
            self.external_heads[base] = nn.Linear(d_model, sz)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = self.type_embedding(batch["input_type_ids"])
        x = x + self.event_description_embedding(batch["input_event_description_ids"])
        x = x + self.group_code_embedding(batch["input_group_code_ids"])
        x = x + self.diagnosis_value_embedding(batch["input_diagnosis_value_ids"])
        x = x + self.gap_embedding(batch["input_gap_ids"])
        x = x + self.setting_embedding(batch["input_setting_ids"])
        x = x + self.dept_type_embedding(batch["input_dept_type_ids"])
        x = x + self.dept_specialty_embedding(batch["input_dept_specialty_ids"])
        x = x + self.facility_size_embedding(batch["input_facility_size_ids"])
        x = x + self.region_embedding(batch["input_region_ids"])
        for base in self.external_stream_bases:
            x = x + self.external_embeddings[base](batch[f"input_{base}_ids"])
        for idx, embedding in enumerate(self.sdoh_embeddings):
            x = x + embedding(batch[f"input_sdoh_{idx}"])
        if len(self.patient_context_embeddings) > 0:
            context_embed = 0
            for idx, embedding in enumerate(self.patient_context_embeddings):
                context_embed = context_embed + embedding(batch["static_context_ids"][:, idx])
            x = x + context_embed.unsqueeze(1)
        if self.patient_numeric_projection is not None:
            numeric_embed = self.patient_numeric_projection(batch["static_numeric_values"])
            x = x + numeric_embed.unsqueeze(1)
        if self.external_per_token_proj is not None and self.external_feature_batch_key in batch:
            x = x + self.external_per_token_proj(batch[self.external_feature_batch_key])
        x = self.positional(x)
        x = self.dropout(x)

        attention_mask = batch["attention_mask"]
        if self.backbone_name == "transformer":
            causal_mask = self._causal_mask(x.size(1), x.device)
            key_padding_mask = attention_mask == 0
            hidden = self.backbone(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        else:
            hidden, _ = self.backbone(x)

        out: dict[str, torch.Tensor] = {
            "type": self.type_head(hidden),
            "event_description": self.event_description_head(hidden),
            "group_code": self.group_code_head(hidden),
            "diagnosis_value": self.diagnosis_value_head(hidden),
            "gap": self.gap_head(hidden),
            "setting": self.setting_head(hidden),
            "dept_type": self.dept_type_head(hidden),
            "dept_specialty": self.dept_specialty_head(hidden),
            "facility_size": self.facility_size_head(hidden),
            "region": self.region_head(hidden),
        }
        for base in self.external_stream_bases:
            out[base] = self.external_heads[base](hidden)
        return out


def _pad_mapping(mapping: Any) -> dict[str, int]:
    base = dict(mapping or {})
    out = dict(SPECIAL_TOKENS)
    next_id = max(out.values()) + 1
    for key, value in sorted(base.items(), key=lambda item: int(item[1])):
        if key in out:
            continue
        int_value = int(value)
        if int_value < next_id:
            int_value = next_id
        out[key] = int_value
        next_id = max(next_id, int_value + 1)
    return out


def _reindex_stream(values: list[Any], shift: int) -> list[int]:
    return [int(v) + shift for v in values]


def _mapping_size(mapping: dict[str, int]) -> int:
    if not mapping:
        return 0
    return max(int(v) for v in mapping.values()) + 1


def _infer_patient_context_vocab_sizes(patient_context_to_id: dict[str, Any], fields: list[str]) -> list[int]:
    sizes: list[int] = []
    for field in fields:
        mapping = patient_context_to_id.get(field, {}) if isinstance(patient_context_to_id, dict) else {}
        if isinstance(mapping, dict) and mapping:
            sizes.append(max(int(v) for v in mapping.values()) + len(SPECIAL_TOKENS) + 1)
        else:
            sizes.append(1)
    return sizes


def _companion_vocab_path(input_path: Path) -> Path:
    name = input_path.name
    if name == "patient_sequences_with_external_features.pt":
        return input_path.with_name("final_token_format.json")
    if name.startswith("patient_sequences_") and name.endswith(".pt"):
        suffix = name[len("patient_sequences_") : -len(".pt")]
        return input_path.with_name(f"sequence_model_vocab_{suffix}.json")
    return input_path.with_suffix(".json")


def _load_vocab_payload(vocab_path: Path) -> dict[str, Any]:
    if not vocab_path.exists():
        return {}
    return json.loads(vocab_path.read_text(encoding="utf-8"))


def _external_rows_to_matrix(raw: Any, seq_len: int, ext_dim: int) -> list[list[float]]:
    """Convert sequence-level external_feature_tensor payload to ``seq_len`` rows of length ``ext_dim``."""
    out = [[0.0] * ext_dim for _ in range(seq_len)]
    if raw is None or ext_dim <= 0 or seq_len <= 0:
        return out
    if isinstance(raw, torch.Tensor):
        t = raw.detach().float().cpu()
        if t.dim() == 2 and t.size(-1) == ext_dim:
            for i in range(min(seq_len, t.size(0))):
                for j in range(ext_dim):
                    out[i][j] = float(t[i, j].item())
        return out
    if isinstance(raw, list):
        for i in range(min(seq_len, len(raw))):
            row = raw[i]
            if isinstance(row, torch.Tensor):
                row = row.flatten().tolist()
            if isinstance(row, (list, tuple)):
                for j in range(min(ext_dim, len(row))):
                    try:
                        out[i][j] = float(row[j])
                    except (TypeError, ValueError):
                        pass
    return out


def _first_available(payloads: list[dict[str, Any]], key: str, default: Any = None) -> Any:
    """Resolve a key from PT/vocab dicts, including nested ``metadata`` (as in ``final_token_format.json``)."""
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if key in payload and payload[key] is not None:
            return payload[key]
        meta = payload.get("metadata")
        if isinstance(meta, dict) and key in meta and meta[key] is not None:
            return meta[key]
    return default


def load_prepared_pt(input_path: Path, vocab_path: Path | None = None) -> PreparedData:
    raw = torch.load(input_path, map_location="cpu")
    if not isinstance(raw, dict) or "sequences" not in raw:
        raise SystemExit(f"Unexpected PT artifact format: {input_path}")

    vocab_payload = _load_vocab_payload(vocab_path or _companion_vocab_path(input_path))
    payloads = [raw, vocab_payload]

    sequences_raw = raw.get("sequences") or []
    if not isinstance(sequences_raw, list):
        raise SystemExit("PT artifact has invalid sequences payload.")

    type_to_id = _pad_mapping(_first_available(payloads, "type_to_id", {}))
    event_description_to_id = _pad_mapping(_first_available(payloads, "event_description_to_id", {}))
    group_code_to_id = _pad_mapping(_first_available(payloads, "group_code_to_id", {}))
    diagnosis_value_to_id = _pad_mapping(_first_available(payloads, "diagnosis_value_to_id", {}))
    gap_to_id = _pad_mapping(_first_available(payloads, "gap_to_id", {}))
    setting_to_id = _pad_mapping(_first_available(payloads, "setting_to_id", {}))
    dept_type_to_id = _pad_mapping(_first_available(payloads, "dept_type_to_id", {}))
    dept_specialty_to_id = _pad_mapping(_first_available(payloads, "dept_specialty_to_id", {}))
    facility_size_to_id = _pad_mapping(_first_available(payloads, "facility_size_to_id", {}))
    region_to_id = _pad_mapping(_first_available(payloads, "region_to_id", {}))
    sdoh_status_to_id = _pad_mapping(
        _first_available(
            payloads,
            "sdoh_status_to_id",
            {"UNKNOWN_NOT_YET_MEASURED": 0, "NEGATIVE_NO_NEED": 1, "POSITIVE_NEED_OR_RISK": 2, "OTHER_DECLINED_UNABLE_UNSPECIFIED": 3},
        )
    )
    patient_context_to_id = dict(_first_available(payloads, "patient_context_to_id", {}))

    sample = next((seq for seq in sequences_raw if isinstance(seq, dict)), None)
    if sample is None:
        raise SystemExit("No valid patient sequences were found in the PT artifact.")

    sdoh_fields = _first_available(payloads, "sdoh_status_fields")
    if not isinstance(sdoh_fields, list) or not sdoh_fields:
        sdoh_fields = sorted(
            key for key in sample.keys() if key.startswith("sdoh_") and key.endswith("_latest_status_ids")
        )
    patient_context_fields = sorted((sample.get("patient_context_ids") or {}).keys())
    ctx_vals = sample.get("patient_context_values") or {}
    numeric_from_vocab = _first_available(payloads, "patient_numeric_fields")
    numeric_from_meta = _first_available(payloads, "patient_context_numeric_fields_found")
    default_numeric = ["patient_lat", "patient_lon", "patient_population"]
    if isinstance(numeric_from_vocab, list) and numeric_from_vocab:
        patient_numeric_fields = [key for key in numeric_from_vocab if key in ctx_vals]
    elif isinstance(numeric_from_meta, list) and numeric_from_meta:
        patient_numeric_fields = [key for key in numeric_from_meta if key in ctx_vals]
    else:
        patient_numeric_fields = [key for key in default_numeric if key in ctx_vals]

    has_ext_num = bool(_first_available(payloads, "has_external_numeric_features"))
    external_feature_key = str(_first_available(payloads, "external_feature_tensor_sequence_key") or "external_feature_tensor")
    external_per_token_dim = int(_first_available(payloads, "external_numeric_features_per_token_dim") or 0)
    raw_ext = sample.get(external_feature_key)
    if raw_ext is not None:
        if isinstance(raw_ext, torch.Tensor) and raw_ext.dim() == 2:
            if external_per_token_dim <= 0:
                external_per_token_dim = int(raw_ext.size(-1))
        elif isinstance(raw_ext, list) and raw_ext and isinstance(raw_ext[0], (list, tuple)):
            if external_per_token_dim <= 0:
                external_per_token_dim = len(raw_ext[0])
    elif has_ext_num and external_per_token_dim <= 0:
        external_per_token_dim = 20

    external_stream_bases = discover_external_stream_bases(sample, sdoh_fields)
    external_to_id: dict[str, dict[str, int]] = {}
    for base in external_stream_bases:
        external_to_id[base] = _pad_mapping(_first_available(payloads, f"{base}_to_id", {}))

    sequences: list[dict[str, Any]] = []
    for seq in sequences_raw:
        if not isinstance(seq, dict):
            continue
        type_ids = _reindex_stream(list(seq.get("type_ids", [])), len(SPECIAL_TOKENS))
        if len(type_ids) < 2:
            continue
        prepared_seq = {
            "patient_id": seq.get("patient_id", "UNKNOWN"),
            "type_ids": type_ids,
            "event_description_ids": _reindex_stream(list(seq.get("event_description_ids", [])), len(SPECIAL_TOKENS)),
            "group_code_ids": _reindex_stream(list(seq.get("group_code_ids", [])), len(SPECIAL_TOKENS)),
            "diagnosis_value_ids": _reindex_stream(list(seq.get("diagnosis_value_ids", [])), len(SPECIAL_TOKENS)),
            "gap_ids": _reindex_stream(list(seq.get("gap_ids", [])), len(SPECIAL_TOKENS)),
            "setting_ids": _reindex_stream(list(seq.get("setting_ids", [])), len(SPECIAL_TOKENS)),
            "dept_type_ids": _reindex_stream(list(seq.get("dept_type_ids", [])), len(SPECIAL_TOKENS)),
            "dept_specialty_ids": _reindex_stream(list(seq.get("dept_specialty_ids", [])), len(SPECIAL_TOKENS)),
            "facility_size_ids": _reindex_stream(list(seq.get("facility_size_ids", [])), len(SPECIAL_TOKENS)),
            "region_ids": _reindex_stream(list(seq.get("region_ids", [])), len(SPECIAL_TOKENS)),
            "patient_context_ids": dict(seq.get("patient_context_ids") or {}),
            "patient_context_values": dict(seq.get("patient_context_values") or {}),
        }
        lengths = {len(prepared_seq[key]) for key in [
            "type_ids",
            "event_description_ids",
            "group_code_ids",
            "diagnosis_value_ids",
            "gap_ids",
            "setting_ids",
            "dept_type_ids",
            "dept_specialty_ids",
            "facility_size_ids",
            "region_ids",
        ]}
        for idx, field in enumerate(sdoh_fields):
            values = _reindex_stream(list(seq.get(field, [])), len(SPECIAL_TOKENS))
            prepared_seq[field] = values
            prepared_seq[f"sdoh_index::{field}"] = idx
            lengths.add(len(values))
        skip_seq = False
        for base in external_stream_bases:
            key_ids = f"{base}_ids"
            raw_list = seq.get(key_ids)
            if not isinstance(raw_list, list):
                skip_seq = True
                break
            values = _reindex_stream(list(raw_list), len(SPECIAL_TOKENS))
            if len(values) != len(type_ids):
                skip_seq = True
                break
            prepared_seq[key_ids] = values
            lengths.add(len(values))
        if skip_seq:
            continue
        if len(lengths) != 1:
            continue
        sequences.append(prepared_seq)

    metadata = dict(_first_available(payloads, "metadata", {}))
    metadata.update(
        {
            "n_training_sequences": int(len(sequences)),
            "type_classes": int(len(type_to_id)),
            "event_description_classes": int(len(event_description_to_id)),
            "group_code_classes": int(len(group_code_to_id)),
            "diagnosis_value_classes": int(len(diagnosis_value_to_id)),
            "gap_classes": int(len(gap_to_id)),
            "setting_classes": int(len(setting_to_id)),
            "dept_type_classes": int(len(dept_type_to_id)),
            "dept_specialty_classes": int(len(dept_specialty_to_id)),
            "facility_size_classes": int(len(facility_size_to_id)),
            "region_classes": int(len(region_to_id)),
            "sdoh_stream_count": int(len(sdoh_fields)),
            "patient_context_field_count": int(len(patient_context_fields)),
            "patient_numeric_field_count": int(len(patient_numeric_fields)),
            "external_stream_bases": list(external_stream_bases),
            "external_stream_count": int(len(external_stream_bases)),
            "external_per_token_dim": int(external_per_token_dim),
            "external_feature_key": external_feature_key,
            "has_external_numeric_tensor": bool(external_per_token_dim > 0),
        }
    )

    return PreparedData(
        sequences=sequences,
        type_to_id=type_to_id,
        event_description_to_id=event_description_to_id,
        group_code_to_id=group_code_to_id,
        diagnosis_value_to_id=diagnosis_value_to_id,
        gap_to_id=gap_to_id,
        setting_to_id=setting_to_id,
        dept_type_to_id=dept_type_to_id,
        dept_specialty_to_id=dept_specialty_to_id,
        facility_size_to_id=facility_size_to_id,
        region_to_id=region_to_id,
        patient_context_to_id=patient_context_to_id,
        sdoh_status_to_id=sdoh_status_to_id,
        sdoh_fields=sdoh_fields,
        patient_context_fields=patient_context_fields,
        patient_numeric_fields=patient_numeric_fields,
        metadata=metadata,
        external_stream_bases=external_stream_bases,
        external_to_id=external_to_id,
        external_per_token_dim=external_per_token_dim,
        external_feature_key=external_feature_key,
    )


def split_sequences(sequences: list[dict[str, Any]], valid_fraction: float, random_seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(random_seed)
    indices = list(range(len(sequences)))
    rng.shuffle(indices)
    valid_n = max(1, int(len(indices) * valid_fraction)) if len(indices) > 1 else 0
    valid_idx = set(indices[:valid_n])
    train = [seq for i, seq in enumerate(sequences) if i not in valid_idx]
    valid = [seq for i, seq in enumerate(sequences) if i in valid_idx]
    return train, valid


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    position_mask: torch.Tensor | None = None,
    head_to_target: dict[str, str] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy per head; if position_mask is set, average only over masked target steps."""
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    per_head: dict[str, torch.Tensor] = {}
    htm = head_to_target or HEAD_TO_TARGET
    for head_key, target_suffix in htm.items():
        logits = outputs[head_key].transpose(1, 2)
        targets = batch[f"target_{target_suffix}"]
        if position_mask is None:
            per_head[head_key] = ce(logits, targets)
        else:
            per_tok = F.cross_entropy(logits, targets, ignore_index=-100, reduction="none")
            m = position_mask & (targets != -100)
            if m.any():
                per_head[head_key] = per_tok[m].mean()
            else:
                per_head[head_key] = outputs[head_key].float().sum() * 0.0
    total = sum(weights[k] * per_head[k] for k in weights if k in per_head)
    metrics = {k: float(per_head[k].detach().cpu()) for k in per_head}
    metrics["total"] = float(total.detach().cpu())
    return total, metrics


def _accumulate_accuracy_micro(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    position_mask: torch.Tensor,
    top_k: int,
    correct_sum: dict[str, float],
    total_sum: dict[str, float],
    topk_correct_sum: dict[str, float],
    head_to_target: dict[str, str],
) -> None:
    for head_key, target_suffix in head_to_target.items():
        logits = outputs[head_key]
        targets = batch[f"target_{target_suffix}"]
        m = position_mask & (targets != -100)
        if not m.any():
            continue
        pred = logits.argmax(dim=1)
        correct = (pred == targets) & m
        ck = f"acc_{head_key}"
        correct_sum[ck] = correct_sum.get(ck, 0.0) + correct.sum().float().item()
        total_sum[ck] = total_sum.get(ck, 0.0) + m.sum().float().item()
        if top_k > 1 and logits.size(1) >= top_k:
            kk = min(top_k, logits.size(1))
            _, topv = logits.topk(kk, dim=1)
            hit = (topv == targets.unsqueeze(1)).any(dim=1) & m
            tk = f"top{top_k}_{head_key}"
            topk_correct_sum[tk] = topk_correct_sum.get(tk, 0.0) + hit.sum().float().item()
            total_sum[tk] = total_sum.get(tk, 0.0) + m.sum().float().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    weights: dict[str, float],
    *,
    head_to_target: dict[str, str] | None = None,
    temporal: bool = False,
    split_role_filter: int | None = None,
    compute_accuracy: bool = False,
    top_k: int = 1,
    progress_desc: str | None = None,
    show_progress: bool = True,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals: dict[str, float] = {}
    correct_sum: dict[str, float] = {}
    total_sum: dict[str, float] = {}
    topk_correct_sum: dict[str, float] = {}
    htm = head_to_target or HEAD_TO_TARGET
    batches = 0
    iterator: Iterable[Any] = loader
    total_batches = len(loader)
    use_tqdm = bool(show_progress and progress_desc and tqdm is not None and total_batches > 0)
    if use_tqdm:
        iterator = tqdm(
            loader,
            total=total_batches,
            desc=progress_desc,
            leave=False,
            unit="batch",
            mininterval=0.3,
            ncols=120,
        )
    t0 = time.perf_counter()
    for batch in iterator:
        batch = {k: v.to(device) for k, v in batch.items()}
        position_mask: torch.Tensor | None = None
        if temporal:
            assert split_role_filter is not None
            position_mask = batch["split_role"] == split_role_filter
        with torch.set_grad_enabled(train_mode):
            outputs = model(batch)
            loss, metrics = compute_losses(outputs, batch, weights, position_mask, head_to_target=htm)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        if compute_accuracy and temporal and split_role_filter is not None:
            pm = batch["split_role"] == split_role_filter
            _accumulate_accuracy_micro(outputs, batch, pm, top_k, correct_sum, total_sum, topk_correct_sum, htm)
        batches += 1
        if use_tqdm and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{metrics.get('total', float('nan')):.4f}")
    elapsed = time.perf_counter() - t0
    label = progress_desc or ("train" if train_mode else "eval")
    if batches == 0:
        log.warning("%s: no batches (empty loader)", label)
        return totals
    out = {key: value / batches for key, value in totals.items()}
    for ck, c in correct_sum.items():
        denom = total_sum.get(ck, 0.0)
        if denom > 0:
            out[ck] = c / denom
    for tk, c in topk_correct_sum.items():
        denom = total_sum.get(tk, 0.0)
        if denom > 0:
            out[tk] = c / denom
    mean_total = out.get("total", float("nan"))
    log.info(
        "%s | batches=%d | wall_time=%.1fs | mean_total_loss=%.6f",
        label,
        batches,
        elapsed,
        mean_total,
    )
    return out


def save_artifacts(output_dir: Path, prepared: PreparedData, model: nn.Module, history: list[dict[str, Any]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "metadata": prepared.metadata}, output_dir / "patient_event_model.pt")
    artifact = {
        "metadata": prepared.metadata,
        "type_to_id": prepared.type_to_id,
        "event_description_to_id": prepared.event_description_to_id,
        "group_code_to_id": prepared.group_code_to_id,
        "diagnosis_value_to_id": prepared.diagnosis_value_to_id,
        "gap_to_id": prepared.gap_to_id,
        "setting_to_id": prepared.setting_to_id,
        "dept_type_to_id": prepared.dept_type_to_id,
        "dept_specialty_to_id": prepared.dept_specialty_to_id,
        "facility_size_to_id": prepared.facility_size_to_id,
        "region_to_id": prepared.region_to_id,
        "patient_context_to_id": prepared.patient_context_to_id,
        "sdoh_status_to_id": prepared.sdoh_status_to_id,
        "sdoh_fields": prepared.sdoh_fields,
        "patient_context_fields": prepared.patient_context_fields,
        "patient_numeric_fields": prepared.patient_numeric_fields,
        "external_stream_bases": prepared.external_stream_bases,
        "training_history": history,
    }
    for base in prepared.external_stream_bases:
        artifact[f"{base}_to_id"] = dict(prepared.external_to_id.get(base, {}))
    (output_dir / "patient_event_model_artifacts.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a multi-task autoregressive patient event sequence model.")
    p.add_argument("--input", type=Path, default=Path("token_sequence_model/patient_sequences_with_external_features.pt"))
    p.add_argument(
        "--vocab-json",
        type=Path,
        default=Path("token_sequence_model/final_token_format.json"),
        help="Full vocab + metadata (internal maps and external-feature specs); default matches patient_sequences_with_external_features.pt.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("data/processed/patient_event_model"))
    p.add_argument("--backbone", choices=["transformer", "gru", "lstm"], default="transformer")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--split-strategy",
        choices=["temporal", "patient_holdout"],
        default="temporal",
        help="temporal: train/val/test targets within each patient sequence; patient_holdout: hold out entire patient sequences for validation.",
    )
    p.add_argument("--temporal-train-frac", type=float, default=0.7, help="Fraction of events (per sequence) used as train targets.")
    p.add_argument("--temporal-val-frac", type=float, default=0.15, help="Fraction of events for validation targets.")
    p.add_argument("--temporal-test-frac", type=float, default=0.15, help="Fraction of events for test targets.")
    p.add_argument(
        "--valid-fraction",
        type=float,
        default=0.1,
        help="patient_holdout only: fraction of patient sequences held out for validation.",
    )
    p.add_argument("--top-k-accuracy", type=int, default=1, help="If >1, report top-k hit rate on val/test (temporal split).")
    p.add_argument("--min-token-freq", type=int, default=1)
    p.add_argument("--num-regions", type=int, default=16)
    p.add_argument("--w-gap", type=float, default=1.0)
    p.add_argument("--w-setting", type=float, default=1.0)
    p.add_argument("--w-dept", type=float, default=1.0)
    p.add_argument("--w-dept-specialty", type=float, default=1.0)
    p.add_argument("--w-size", type=float, default=1.0)
    p.add_argument("--w-region", type=float, default=1.0)
    p.add_argument("--w-type", type=float, default=1.0)
    p.add_argument("--w-event-description", type=float, default=1.0)
    p.add_argument("--w-group-code", type=float, default=1.0)
    p.add_argument("--w-diagnosis-value", type=float, default=1.0)
    p.add_argument(
        "--w-external",
        type=float,
        default=1.0,
        help="Loss weight for each auto-discovered external per-timestep stream (<base>_ids in the .pt).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm batch progress bars (still logs per-phase summaries).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for human-readable lines (epoch summaries use log.info).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    if tqdm is None:
        log.warning("tqdm is not installed; batch-level progress bars disabled. pip install tqdm")

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")
    if args.vocab_json and not args.vocab_json.exists():
        inferred_vocab = _companion_vocab_path(args.input)
        if inferred_vocab.exists():
            args.vocab_json = inferred_vocab
        else:
            raise SystemExit(f"Vocab JSON not found: {args.vocab_json}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.input.suffix != ".pt":
        raise SystemExit("This training script now expects the final tokenized .pt artifact as --input.")
    prepared = load_prepared_pt(args.input, args.vocab_json)
    if not prepared.sequences:
        raise SystemExit("No patient sequences with at least two events were found.")

    head_to_target = merge_head_to_target(prepared.external_stream_bases)
    external_vocab_sizes = {b: _mapping_size(prepared.external_to_id[b]) for b in prepared.external_stream_bases}
    if prepared.external_stream_bases:
        log.info(
            "External categorical streams: %s",
            json.dumps(
                {"bases": prepared.external_stream_bases, "vocab_sizes": external_vocab_sizes},
            ),
        )

    sdoh_inputs = [f"sdoh_{idx}" for idx in range(len(prepared.sdoh_fields))]

    if args.split_strategy == "temporal":
        tf, vf, sf = args.temporal_train_frac, args.temporal_val_frac, args.temporal_test_frac
        s = tf + vf + sf
        if s <= 0 or tf < 0 or vf < 0 or sf < 0:
            raise SystemExit(f"Temporal fractions must be non-negative and sum to a positive value; got train={tf}, val={vf}, test={sf}.")
        tf, vf, sf = tf / s, vf / s, sf / s
        indexed_all = []
        for seq in prepared.sequences:
            clone = dict(seq)
            for idx, field in enumerate(prepared.sdoh_fields):
                clone[f"sdoh_{idx}"] = clone[field]
            indexed_all.append(clone)
        full_ds = PatientSequenceDataset(
            indexed_all,
            max_seq_len=args.max_seq_len,
            sdoh_fields=sdoh_inputs,
            patient_context_fields=prepared.patient_context_fields,
            patient_numeric_fields=prepared.patient_numeric_fields,
            temporal_split=True,
            train_fraction=tf,
            valid_fraction=vf,
            test_fraction=sf,
            external_stream_bases=prepared.external_stream_bases,
            external_per_token_dim=prepared.external_per_token_dim,
            external_feature_key=prepared.external_feature_key,
        )
        train_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True)
        eval_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False)
        valid_loader = eval_loader
        use_temporal = True
    else:
        train_sequences, valid_sequences = split_sequences(prepared.sequences, args.valid_fraction, args.seed)
        if not train_sequences:
            train_sequences = prepared.sequences
            valid_sequences = []

        indexed_train_sequences = []
        for seq in train_sequences:
            clone = dict(seq)
            for idx, field in enumerate(prepared.sdoh_fields):
                clone[f"sdoh_{idx}"] = clone[field]
            indexed_train_sequences.append(clone)
        indexed_valid_sequences = []
        for seq in valid_sequences:
            clone = dict(seq)
            for idx, field in enumerate(prepared.sdoh_fields):
                clone[f"sdoh_{idx}"] = clone[field]
            indexed_valid_sequences.append(clone)

        train_ds = PatientSequenceDataset(
            indexed_train_sequences,
            max_seq_len=args.max_seq_len,
            sdoh_fields=sdoh_inputs,
            patient_context_fields=prepared.patient_context_fields,
            patient_numeric_fields=prepared.patient_numeric_fields,
            temporal_split=False,
            external_stream_bases=prepared.external_stream_bases,
            external_per_token_dim=prepared.external_per_token_dim,
            external_feature_key=prepared.external_feature_key,
        )
        valid_ds = (
            PatientSequenceDataset(
                indexed_valid_sequences,
                max_seq_len=args.max_seq_len,
                sdoh_fields=sdoh_inputs,
                patient_context_fields=prepared.patient_context_fields,
                patient_numeric_fields=prepared.patient_numeric_fields,
                temporal_split=False,
                external_stream_bases=prepared.external_stream_bases,
                external_per_token_dim=prepared.external_per_token_dim,
                external_feature_key=prepared.external_feature_key,
            )
            if indexed_valid_sequences
            else None
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False) if valid_ds is not None else None
        eval_loader = valid_loader
        use_temporal = False

    show_progress = not args.no_progress and tqdm is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        "Run configuration | input=%s vocab=%s device=%s sequences=%s split=%s epochs=%s batch_size=%s lr=%s backbone=%s d_model=%s",
        args.input,
        args.vocab_json,
        device,
        f"{len(prepared.sequences):,}",
        args.split_strategy,
        args.epochs,
        args.batch_size,
        args.lr,
        args.backbone,
        args.d_model,
    )
    log.info(
        "External per-token tensor dim=%s key=%s | SDOH streams=%s context_fields=%s numeric_fields=%s",
        prepared.external_per_token_dim,
        prepared.external_feature_key,
        len(prepared.sdoh_fields),
        len(prepared.patient_context_fields),
        prepared.patient_numeric_fields,
    )

    model = PatientEventSequenceModel(
        type_size=_mapping_size(prepared.type_to_id),
        event_description_size=_mapping_size(prepared.event_description_to_id),
        group_code_size=_mapping_size(prepared.group_code_to_id),
        diagnosis_value_size=_mapping_size(prepared.diagnosis_value_to_id),
        gap_size=_mapping_size(prepared.gap_to_id),
        setting_size=_mapping_size(prepared.setting_to_id),
        dept_type_size=_mapping_size(prepared.dept_type_to_id),
        dept_specialty_size=_mapping_size(prepared.dept_specialty_to_id),
        facility_size_size=_mapping_size(prepared.facility_size_to_id),
        region_size=_mapping_size(prepared.region_to_id),
        sdoh_size=_mapping_size(prepared.sdoh_status_to_id),
        n_sdoh_streams=len(prepared.sdoh_fields),
        patient_context_vocab_sizes=_infer_patient_context_vocab_sizes(
            prepared.patient_context_to_id,
            prepared.patient_context_fields,
        ),
        patient_numeric_dim=len(prepared.patient_numeric_fields),
        d_model=args.d_model,
        max_seq_len=args.max_seq_len,
        backbone=args.backbone,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        pad_token_id=SPECIAL_TOKENS["[PAD]"],
        external_stream_bases=prepared.external_stream_bases,
        external_vocab_sizes=external_vocab_sizes,
        external_per_token_dim=prepared.external_per_token_dim,
        external_feature_key=prepared.external_feature_key,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: total=%s trainable=%s heads=%s", f"{n_params:,}", f"{n_trainable:,}", len(head_to_target))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    weights = {
        "type": args.w_type,
        "event_description": args.w_event_description,
        "group_code": args.w_group_code,
        "diagnosis_value": args.w_diagnosis_value,
        "gap": args.w_gap,
        "setting": args.w_setting,
        "dept_type": args.w_dept,
        "dept_specialty": args.w_dept_specialty,
        "facility_size": args.w_size,
        "region": args.w_region,
    }
    for base in prepared.external_stream_bases:
        weights[base] = args.w_external

    log.info(
        "DataLoaders | train_batches=%s eval_batches=%s (batch_size=%s)",
        len(train_loader),
        len(eval_loader) if eval_loader is not None else 0,
        args.batch_size,
    )

    history: list[dict[str, Any]] = []
    top_k = max(1, int(args.top_k_accuracy))

    for epoch in range(1, args.epochs + 1):
        log.info("---------- Epoch %s / %s ----------", epoch, args.epochs)
        if use_temporal:
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                weights,
                head_to_target=head_to_target,
                temporal=True,
                split_role_filter=SPLIT_TRAIN,
                progress_desc=f"[{epoch}/{args.epochs}] train",
                show_progress=show_progress,
            )
            valid_metrics = run_epoch(
                model,
                eval_loader,
                None,
                device,
                weights,
                head_to_target=head_to_target,
                temporal=True,
                split_role_filter=SPLIT_VAL,
                compute_accuracy=True,
                top_k=top_k,
                progress_desc=f"[{epoch}/{args.epochs}] valid",
                show_progress=show_progress,
            )
            test_metrics = run_epoch(
                model,
                eval_loader,
                None,
                device,
                weights,
                head_to_target=head_to_target,
                temporal=True,
                split_role_filter=SPLIT_TEST,
                compute_accuracy=True,
                top_k=top_k,
                progress_desc=f"[{epoch}/{args.epochs}] test",
                show_progress=show_progress,
            )
            record = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics, "test": test_metrics}
            log.info(
                "Epoch %s/%s done | train_loss=%.6f valid_loss=%.6f test_loss=%.6f",
                epoch,
                args.epochs,
                train_metrics.get("total", float("nan")),
                valid_metrics.get("total", float("nan")),
                test_metrics.get("total", float("nan")),
            )
        else:
            train_metrics = run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                weights,
                head_to_target=head_to_target,
                temporal=False,
                progress_desc=f"[{epoch}/{args.epochs}] train",
                show_progress=show_progress,
            )
            valid_metrics = (
                run_epoch(
                    model,
                    valid_loader,
                    None,
                    device,
                    weights,
                    head_to_target=head_to_target,
                    temporal=False,
                    progress_desc=f"[{epoch}/{args.epochs}] valid",
                    show_progress=show_progress,
                )
                if valid_loader is not None
                else None
            )
            record = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics, "test": None}
            vl = valid_metrics.get("total") if valid_metrics else None
            log.info(
                "Epoch %s/%s done | train_loss=%.6f valid_loss=%s",
                epoch,
                args.epochs,
                train_metrics.get("total", float("nan")),
                f"{vl:.6f}" if vl is not None else "n/a",
            )

        history.append(record)
        print(json.dumps(record), flush=True)

    save_artifacts(args.output_dir, prepared, model, history, args)
    log.info("Saved artifacts to %s", args.output_dir.resolve())
    print(json.dumps({"status": "done", "output_dir": str(args.output_dir), "metadata": prepared.metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
