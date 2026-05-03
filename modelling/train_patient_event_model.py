#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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

        # Causal alignment: at time t, inputs use BOS (t==0) or prior-step ids; targets[t] is the token to predict.
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
        # Sum of many embeddings has high variance; stabilize before backbone + heads.
        self.embed_norm = nn.LayerNorm(d_model)
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
                norm_first=True,
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
        # Bool tensor: True = cannot attend (future positions). Matches bool src_key_padding_mask (PyTorch 2.x).
        return torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)

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
        x = self.embed_norm(x)
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


def _load_sequence_artifact_dict(input_path: Path, *, mmap: bool) -> dict[str, Any]:
    """Load the pickle dict from disk.

    When ``mmap=True`` (default), PyTorch may memory-map tensor storages instead of fully
    materializing them in RAM—often helpful for multi‑GB ``.pt`` files. Legacy pickle layouts
    may ignore mmap; then we fall back to a normal load. True chunked/streaming load without
    reading the whole archive requires exporting **sharded** ``.pt`` files (separate pipeline).
    """
    if mmap:
        try:
            raw = torch.load(input_path, map_location="cpu", weights_only=False, mmap=True)
            log.info("Sequence artifact opened with tensor memory-mapping (mmap=True).")
            return raw
        except TypeError:
            try:
                raw = torch.load(input_path, map_location="cpu", mmap=True)
                log.info("Sequence artifact opened with tensor memory-mapping (mmap=True).")
                return raw
            except TypeError:
                pass
        except (RuntimeError, OSError, ValueError) as e:
            log.warning("mmap load failed (%s); falling back to full in-RAM deserialization.", e)
    try:
        raw = torch.load(input_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(input_path, map_location="cpu")
    log.info("Sequence artifact loaded with full in-RAM deserialization (mmap not used).")
    return raw


def load_prepared_pt(input_path: Path, vocab_path: Path | None = None, *, mmap_load: bool = True) -> PreparedData:
    log.info(
        "Loading sequence artifact (large .pt files can take minutes before other logs appear): %s",
        input_path.resolve(),
    )
    raw = _load_sequence_artifact_dict(input_path, mmap=mmap_load)
    if not isinstance(raw, dict) or "sequences" not in raw:
        raise SystemExit(f"Unexpected PT artifact format: {input_path}")

    n_raw = len(raw.get("sequences") or [])
    log.info("Deserialized %s raw sequence entries; loading vocab and building training sequences…", f"{n_raw:,}")

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

    prepared = PreparedData(
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
    del raw
    gc.collect()
    log.info(
        "Released raw PT dict from RAM; holding %s prepared sequences for training.",
        f"{len(sequences):,}",
    )
    return prepared


def split_sequences(sequences: list[dict[str, Any]], valid_fraction: float, random_seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(random_seed)
    indices = list(range(len(sequences)))
    rng.shuffle(indices)
    valid_n = max(1, int(len(indices) * valid_fraction)) if len(indices) > 1 else 0
    valid_idx = set(indices[:valid_n])
    train = [seq for i, seq in enumerate(sequences) if i not in valid_idx]
    valid = [seq for i, seq in enumerate(sequences) if i in valid_idx]
    return train, valid


def _assert_logits_targets_aligned(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    head_to_target: dict[str, str],
) -> None:
    """``outputs[head][b, t, :]`` predicts ``batch['target_*'][b, t]`` (same ``t`` as dataset targets).

    The dataset builds ``input_*[:, t]`` as BOS (t=0) or the previous step's token so that the
    causal stack at time ``t`` forecasts the current-step targets without an off-by-one.
    """
    for head_key, target_suffix in head_to_target.items():
        if head_key not in outputs:
            continue
        logits = outputs[head_key]
        tk = f"target_{target_suffix}"
        if tk not in batch:
            raise KeyError(f"Missing batch column {tk!r} for head {head_key!r}")
        targets = batch[tk]
        if logits.shape[:2] != targets.shape:
            raise ValueError(
                f"{head_key}: logits shape {tuple(logits.shape)} vs targets {tuple(targets.shape)} — "
                "expected logits [batch, time, classes] and targets [batch, time]"
            )


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float],
    position_mask: torch.Tensor | None = None,
    head_to_target: dict[str, str] | None = None,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-head cross-entropy.

    ``F.cross_entropy(logits, targets)`` expects ``logits`` shaped ``[B, num_classes, T]``; we pass
    ``outputs[head].transpose(1, 2)`` so class logits at ``[b, :, t]`` match integer target ``targets[b, t]``.

    With ``position_mask`` (temporal split), loss is averaged only where the mask is True and
    ``targets != -100`` (padding).
    """
    htm = head_to_target or HEAD_TO_TARGET
    _assert_logits_targets_aligned(outputs, batch, htm)
    if position_mask is not None:
        ref_shape = None
        for _, suf in htm.items():
            tk = f"target_{suf}"
            if tk in batch:
                ref_shape = batch[tk].shape
                break
        if ref_shape is not None and position_mask.shape != ref_shape:
            raise ValueError(
                f"position_mask shape {tuple(position_mask.shape)} != target shape {tuple(ref_shape)}"
            )
    ce = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
    per_head: dict[str, torch.Tensor] = {}
    ce_kw: dict[str, Any] = {"ignore_index": -100, "reduction": "none"}
    if label_smoothing > 0.0:
        ce_kw["label_smoothing"] = label_smoothing
    for head_key, target_suffix in htm.items():
        logits = outputs[head_key].transpose(1, 2)
        targets = batch[f"target_{target_suffix}"]
        if position_mask is None:
            per_head[head_key] = ce(logits, targets)
        else:
            per_tok = F.cross_entropy(logits, targets, **ce_kw)
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
        # Logits are (B, T, num_classes); argmax over class dim matches target (B, T).
        pred = logits.argmax(dim=-1)
        correct = (pred == targets) & m
        ck = f"acc_{head_key}"
        correct_sum[ck] = correct_sum.get(ck, 0.0) + correct.sum().float().item()
        total_sum[ck] = total_sum.get(ck, 0.0) + m.sum().float().item()
        n_cls = logits.size(-1)
        if top_k > 1 and n_cls >= top_k:
            kk = min(top_k, n_cls)
            _, topv = logits.topk(kk, dim=-1)
            hit = (topv == targets.unsqueeze(-1)).any(dim=-1) & m
            tk = f"top{top_k}_{head_key}"
            topk_correct_sum[tk] = topk_correct_sum.get(tk, 0.0) + hit.sum().float().item()
            total_sum[tk] = total_sum.get(tk, 0.0) + m.sum().float().item()


def _finalize_split_metrics(
    totals: dict[str, float],
    batches: int,
    correct_sum: dict[str, float],
    total_sum: dict[str, float],
    topk_correct_sum: dict[str, float],
) -> dict[str, float]:
    """Merge batch-accumulated loss means with token-micro accuracy.

    Per-head / total **CE** here is the mean of per-batch means (each batch’s loss is already
    mean over **active tokens in that batch** for that head). Long and short sequences therefore
    weight equally **per batch**, not per token. **acc_** / **topk_** keys are correct/total over
    all active tokens in the pass (micro-averaged).
    """
    out = {key: totals[key] / batches for key in totals}
    for ck, c in correct_sum.items():
        denom = total_sum.get(ck, 0.0)
        if denom > 0:
            out[ck] = c / denom
    for tk, c in topk_correct_sum.items():
        denom = total_sum.get(tk, 0.0)
        if denom > 0:
            out[tk] = c / denom
    return out


def log_per_head_mean_ce(phase_label: str, metrics: dict[str, float], head_weight_order: list[str]) -> None:
    """Log one line per phase with mean CE per head (same keys as loss weights) plus total."""
    parts: list[str] = []
    for hk in head_weight_order:
        if hk in metrics:
            parts.append(f"{hk}={metrics[hk]:.4f}")
    log.info(
        "%s | per-head mean CE: %s | total=%.6f",
        phase_label,
        " ".join(parts) if parts else "(no heads)",
        metrics.get("total", float("nan")),
    )


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
    log_batch_interval: int = 0,
    label_smoothing: float = 0.0,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals: dict[str, float] = {}
    correct_sum: dict[str, float] = {}
    total_sum: dict[str, float] = {}
    topk_correct_sum: dict[str, float] = {}
    htm = head_to_target or HEAD_TO_TARGET
    batches = 0
    running_total_loss = 0.0
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
            loss, metrics = compute_losses(
                outputs,
                batch,
                weights,
                position_mask,
                head_to_target=htm,
                label_smoothing=label_smoothing,
            )
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip and grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        if compute_accuracy and temporal and split_role_filter is not None:
            pm = batch["split_role"] == split_role_filter
            _accumulate_accuracy_micro(outputs, batch, pm, top_k, correct_sum, total_sum, topk_correct_sum, htm)
        batches += 1
        mt = float(metrics.get("total", 0.0))
        running_total_loss += mt
        if log_batch_interval > 0 and batches % log_batch_interval == 0:
            phase = progress_desc or ("train" if train_mode else "eval")
            log.info(
                "%s | batches %s/%s | running_mean_total_loss=%.6f | last_batch_total=%.6f",
                phase,
                batches,
                total_batches,
                running_total_loss / batches,
                mt,
            )
        if use_tqdm and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{metrics.get('total', float('nan')):.4f}")
    elapsed = time.perf_counter() - t0
    label = progress_desc or ("train" if train_mode else "eval")
    if batches == 0:
        log.warning("%s: no batches (empty loader)", label)
        return totals
    out = _finalize_split_metrics(totals, batches, correct_sum, total_sum, topk_correct_sum)
    mean_total = out.get("total", float("nan"))
    log.info(
        "%s | batches=%d | wall_time=%.1fs | mean_total_loss=%.6f",
        label,
        batches,
        elapsed,
        mean_total,
    )
    return out


def run_eval_temporal_val_test_combined(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    *,
    head_to_target: dict[str, str] | None = None,
    top_k: int = 1,
    progress_desc: str | None = None,
    show_progress: bool = True,
    log_batch_interval: int = 0,
    label_smoothing: float = 0.0,
) -> tuple[dict[str, float], dict[str, float]]:
    """Single forward pass per batch; val and test losses use different split_role masks.

    Cuts temporal eval compute roughly in half vs separate valid + test passes over the same loader.
    """
    model.eval()
    htm = head_to_target or HEAD_TO_TARGET
    totals_v: dict[str, float] = {}
    totals_t: dict[str, float] = {}
    correct_v: dict[str, float] = {}
    total_v: dict[str, float] = {}
    topkv: dict[str, float] = {}
    correct_t: dict[str, float] = {}
    total_t: dict[str, float] = {}
    topkt: dict[str, float] = {}
    batches = 0
    running_val_total = 0.0
    running_test_total = 0.0
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
    label = progress_desc or "eval_val_test"
    for batch in iterator:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch)
            m_val = batch["split_role"] == SPLIT_VAL
            m_test = batch["split_role"] == SPLIT_TEST
            _, met_v = compute_losses(
                outputs,
                batch,
                weights,
                m_val,
                head_to_target=htm,
                label_smoothing=label_smoothing,
            )
            _, met_t = compute_losses(
                outputs,
                batch,
                weights,
                m_test,
                head_to_target=htm,
                label_smoothing=label_smoothing,
            )
        for k, v in met_v.items():
            totals_v[k] = totals_v.get(k, 0.0) + float(v)
        for k, v in met_t.items():
            totals_t[k] = totals_t.get(k, 0.0) + float(v)
        _accumulate_accuracy_micro(outputs, batch, m_val, top_k, correct_v, total_v, topkv, htm)
        _accumulate_accuracy_micro(outputs, batch, m_test, top_k, correct_t, total_t, topkt, htm)
        batches += 1
        running_val_total += float(met_v.get("total", 0.0))
        running_test_total += float(met_t.get("total", 0.0))
        if log_batch_interval > 0 and batches % log_batch_interval == 0:
            log.info(
                "%s | batches %s/%s | running_mean_val_total=%.6f | running_mean_test_total=%.6f",
                label,
                batches,
                total_batches,
                running_val_total / batches,
                running_test_total / batches,
            )
        if use_tqdm and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                val_tot=f"{float(met_v.get('total', 0.0)):.4f}",
                test_tot=f"{float(met_t.get('total', 0.0)):.4f}",
            )
    elapsed = time.perf_counter() - t0
    if batches == 0:
        log.warning("%s: no batches (empty loader)", label)
        return {}, {}
    out_v = _finalize_split_metrics(totals_v, batches, correct_v, total_v, topkv)
    out_t = _finalize_split_metrics(totals_t, batches, correct_t, total_t, topkt)
    log.info(
        "%s | batches=%d | wall_time=%.1fs | val_mean_total=%.6f | test_mean_total=%.6f (single pass)",
        label,
        batches,
        elapsed,
        out_v.get("total", float("nan")),
        out_t.get("total", float("nan")),
    )
    return out_v, out_t


def reset_incremental_metric_files(output_dir: Path) -> None:
    """Start a fresh JSONL log for this training process."""
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "training_metrics.jsonl"
    if p.exists():
        p.unlink()


def append_epoch_metrics_jsonl(output_dir: Path, record: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def save_training_history_snapshot(output_dir: Path, history: list[dict[str, Any]]) -> None:
    """Full metrics list so far — safe to plot if training is interrupted."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_history_snapshot.json").write_text(
        json.dumps(history, indent=2, default=str),
        encoding="utf-8",
    )


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


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_completed: int,
    history: list[dict[str, Any]],
    args: argparse.Namespace | None = None,
    scheduler: Any | None = None,
) -> None:
    """Training-state checkpoint (not written until an epoch finishes all its run_epoch phases)."""
    payload: dict[str, Any] = {
        "format_version": 1,
        "epoch": epoch_completed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if args is not None:
        payload["args"] = vars(args)
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    _atomic_torch_save(payload, path)


def _restore_torch_rng_state(rs: Any) -> None:
    """Restore PyTorch CPU RNG state; tolerate dtype/device quirks after torch.load."""
    if rs is None:
        return
    try:
        if torch.is_tensor(rs):
            t = rs.detach().cpu().contiguous()
            if t.dtype == torch.uint8:
                torch.set_rng_state(t)
                return
            # torch.load sometimes revives uint8 state as another dtype; reinterpret bytes.
            raw = t.numpy().tobytes()
            torch.set_rng_state(torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone())
            return
        arr = np.asarray(rs)
        if arr.dtype != np.uint8:
            arr = np.frombuffer(arr.tobytes(), dtype=np.uint8)
        torch.set_rng_state(torch.from_numpy(np.ascontiguousarray(arr)))
    except Exception as exc:
        log.warning(
            "Could not restore torch RNG state from checkpoint (%s); continuing without RNG restore.",
            exc,
        )


def load_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Any | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch_done = int(ckpt.get("epoch", 0))
    history = list(ckpt.get("history", []))
    sd_sched = ckpt.get("scheduler_state_dict")
    sched_loaded = False
    if scheduler is not None:
        if isinstance(sd_sched, dict):
            try:
                scheduler.load_state_dict(sd_sched)
                sched_loaded = True
            except Exception as exc:
                log.warning("Could not load LR scheduler state from checkpoint (%s).", exc)
        if not sched_loaded and epoch_done > 0:
            log.info(
                "LR scheduler: applying %s scheduler.step() call(s) to match resumed epoch "
                "(checkpoint has no usable scheduler state).",
                epoch_done,
            )
            for _ in range(epoch_done):
                scheduler.step()
    _restore_torch_rng_state(ckpt.get("torch_rng_state"))
    nprs = ckpt.get("numpy_rng_state")
    if nprs is not None:
        np.random.set_state(nprs)
    pyrs = ckpt.get("python_rng_state")
    if pyrs is not None:
        random.setstate(pyrs)
    return epoch_done, history


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> Any:
    """Per-epoch LR updates (call ``scheduler.step()`` once after each epoch)."""
    if args.lr_scheduler == "none":
        return None
    eta_min = max(0.0, float(args.lr) * float(args.lr_min_ratio))
    epochs = max(1, int(args.epochs))
    wu_raw = max(0, int(args.warmup_epochs))
    wu = min(wu_raw, max(0, epochs - 1))

    if args.lr_scheduler == "cosine_warmup":
        if wu > 0 and epochs > wu:
            warm = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=wu)
            cos_t = max(1, epochs - wu)
            cool = CosineAnnealingLR(optimizer, T_max=cos_t, eta_min=eta_min)
            return SequentialLR(optimizer, schedulers=[warm, cool], milestones=[wu])
        if wu > 0:
            return LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(1, epochs))
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a multi-task autoregressive patient event sequence model.")
    p.add_argument("--input", type=Path, default=Path("token_sequence_model/patient_sequences_with_external_features.pt"))
    p.add_argument(
        "--no-mmap-load",
        action="store_true",
        help="Disable memory-mapped tensor load for the sequence .pt (full RAM deserialize). Default uses mmap when PyTorch supports it to lower peak RAM on large files.",
    )
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
    p.add_argument(
        "--max-sequences",
        type=int,
        default=0,
        help="If >0, train/eval only on the first N sequences after load (deterministic order). Use for smoke tests.",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fast pipeline check: caps sequences (128), epochs (3), batch size (16); writes to patient_event_model_smoke unless --output-dir set.",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay (L2); use 0 to disable.",
    )
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Cross-entropy label smoothing (0 = standard CE). Can improve calibration; may also flatten loss curves.",
    )
    p.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "cosine_warmup"],
        default="cosine_warmup",
        help="Learning-rate schedule: cosine decay per epoch, optionally after linear warmup.",
    )
    p.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Linear LR warmup length when --lr-scheduler cosine_warmup (clamped vs total epochs).",
    )
    p.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.01,
        help="Cosine floor as a fraction of --lr (final LR ~= lr * this ratio).",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (0 disables clipping).",
    )
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
    p.add_argument(
        "--log-batch-interval",
        type=int,
        default=50,
        help="Within each train/valid/test pass: log running mean total loss every N batches (0 disables).",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Append human-readable log lines (same as console) to this file. Parent dirs are created. "
        "Does not capture tqdm bars; use shell redirection for full terminal capture.",
    )
    p.add_argument(
        "--no-incremental-metrics",
        action="store_true",
        help="Do not write training_metrics.jsonl or training_history_snapshot.json after each epoch.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint_last.pt only after a **full epoch**: temporal = train+valid+test passes; "
        "patient_holdout = train+valid (no test split). Same path overwritten each time. 0 = no .pt checkpoints.",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint_last.pt (or any save_training_checkpoint file). Same --input/vocab/architecture. "
        "Set --epochs to the **total** desired epochs (e.g. 10 after a 3-epoch run). "
        "With --no-incremental-metrics unset, training_metrics.jsonl is appended (not wiped); train.log should use the same --log-file path. "
        "Checkpoints saved after this version include LR scheduler state; older checkpoints advance the scheduler by completed epoch count.",
    )
    p.add_argument(
        "--no-save-best",
        action="store_true",
        help="Do not write checkpoint_best.pt when validation total loss improves.",
    )
    ns = p.parse_args()
    _default_out = Path("data/processed/patient_event_model")
    if ns.smoke_test:
        if ns.max_sequences <= 0:
            ns.max_sequences = 128
        ns.epochs = 3
        ns.batch_size = 16
        if ns.output_dir == _default_out:
            ns.output_dir = Path("data/processed/patient_event_model_smoke")
    return ns


def main() -> None:
    args = parse_args()
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)
    if args.log_file is not None:
        log.info("Also writing log lines to %s", args.log_file.resolve())
    if tqdm is None:
        log.warning("tqdm is not installed; batch-level progress bars disabled. pip install tqdm")

    if args.smoke_test:
        log.info(
            "Smoke-test mode | max_sequences=%s epochs=%s batch_size=%s output_dir=%s",
            args.max_sequences,
            args.epochs,
            args.batch_size,
            args.output_dir,
        )

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
    prepared = load_prepared_pt(args.input, args.vocab_json, mmap_load=not args.no_mmap_load)
    if not prepared.sequences:
        raise SystemExit("No patient sequences with at least two events were found.")

    if args.max_sequences > 0:
        cap = min(len(prepared.sequences), args.max_sequences)
        md = dict(prepared.metadata)
        md["sequence_subset_cap"] = cap
        md["sequence_subset_total_loaded"] = len(prepared.sequences)
        prepared = replace(prepared, sequences=prepared.sequences[:cap], metadata=md)
        log.info("Using %s sequences (--max-sequences %s).", cap, args.max_sequences)

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
        "Run configuration | input=%s vocab=%s device=%s sequences=%s split=%s epochs=%s batch_size=%s "
        "lr=%s scheduler=%s weight_decay=%s label_smoothing=%s backbone=%s d_model=%s",
        args.input,
        args.vocab_json,
        device,
        f"{len(prepared.sequences):,}",
        args.split_strategy,
        args.epochs,
        args.batch_size,
        args.lr,
        args.lr_scheduler,
        args.weight_decay,
        args.label_smoothing,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=max(0.0, float(args.weight_decay)),
    )
    scheduler = _build_lr_scheduler(optimizer, args)
    if scheduler is not None:
        log.info(
            "LR schedule: %s | warmup_epochs=%s lr_min≈%.2e (ratio=%s)",
            args.lr_scheduler,
            args.warmup_epochs,
            float(args.lr) * float(args.lr_min_ratio),
            args.lr_min_ratio,
        )
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
    log_batch_iv = max(0, int(args.log_batch_interval))

    if not args.no_incremental_metrics:
        if args.resume is None:
            reset_incremental_metric_files(args.output_dir)
        log.info(
            "Incremental metrics: %s (one JSON object per epoch) and %s (full history after each epoch)%s",
            args.output_dir / "training_metrics.jsonl",
            args.output_dir / "training_history_snapshot.json",
            " — appending to existing files (resume)" if args.resume is not None else "",
        )

    ckpt_last = args.output_dir / "checkpoint_last.pt"
    ckpt_best = args.output_dir / "checkpoint_best.pt"
    best_valid_loss = float("inf")
    start_epoch = 0
    if args.resume is not None:
        if not args.resume.exists():
            raise SystemExit(f"Resume checkpoint not found: {args.resume}")
        start_epoch, history = load_training_checkpoint(
            args.resume, model, optimizer, device, scheduler=scheduler
        )
        log.info(
            "Resumed from %s | last completed epoch=%s | history_len=%s",
            args.resume.resolve(),
            start_epoch,
            len(history),
        )
        for rec in history:
            vm = rec.get("valid")
            if isinstance(vm, dict):
                t = vm.get("total")
                if isinstance(t, (int, float)) and math.isfinite(float(t)):
                    best_valid_loss = min(best_valid_loss, float(t))

    if start_epoch >= args.epochs:
        log.warning(
            "Checkpoint epoch (%s) already >= --epochs (%s); writing final artifacts only.",
            start_epoch,
            args.epochs,
        )
        save_artifacts(args.output_dir, prepared, model, history, args)
        log.info("Saved artifacts to %s", args.output_dir.resolve())
        print(json.dumps({"status": "done", "output_dir": str(args.output_dir), "metadata": prepared.metadata}, indent=2), flush=True)
        return

    last_completed_epoch = start_epoch

    try:
        for epoch in range(start_epoch + 1, args.epochs + 1):
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
                    log_batch_interval=log_batch_iv,
                    label_smoothing=args.label_smoothing,
                    grad_clip=args.grad_clip,
                )
                valid_metrics, test_metrics = run_eval_temporal_val_test_combined(
                    model,
                    eval_loader,
                    device,
                    weights,
                    head_to_target=head_to_target,
                    top_k=top_k,
                    progress_desc=f"[{epoch}/{args.epochs}] valid+test",
                    show_progress=show_progress,
                    log_batch_interval=log_batch_iv,
                    label_smoothing=args.label_smoothing,
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
                head_order = list(weights.keys())
                log_per_head_mean_ce(f"Epoch {epoch}/{args.epochs} train", train_metrics, head_order)
                log_per_head_mean_ce(f"Epoch {epoch}/{args.epochs} valid", valid_metrics, head_order)
                log_per_head_mean_ce(f"Epoch {epoch}/{args.epochs} test", test_metrics, head_order)
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
                    log_batch_interval=log_batch_iv,
                    label_smoothing=args.label_smoothing,
                    grad_clip=args.grad_clip,
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
                        log_batch_interval=log_batch_iv,
                        label_smoothing=args.label_smoothing,
                        grad_clip=args.grad_clip,
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
                head_order = list(weights.keys())
                log_per_head_mean_ce(f"Epoch {epoch}/{args.epochs} train", train_metrics, head_order)
                if valid_metrics:
                    log_per_head_mean_ce(f"Epoch {epoch}/{args.epochs} valid", valid_metrics, head_order)

            if scheduler is not None:
                scheduler.step()
            log.info("Optimizer LR after epoch %s (next epoch): %.2e", epoch, optimizer.param_groups[0]["lr"])

            history.append(record)
            print(json.dumps(record), flush=True)

            if not args.no_incremental_metrics:
                append_epoch_metrics_jsonl(args.output_dir, record)
                save_training_history_snapshot(args.output_dir, history)
                log.info("Wrote incremental metrics (%s epochs in snapshot)", len(history))

            # .pt checkpoints only after full epoch (train → val → test for temporal; train → val for holdout).
            if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
                save_training_checkpoint(
                    ckpt_last, model, optimizer, epoch, history, args, scheduler=scheduler
                )
                log.info("Checkpoint saved after full epoch %s: %s", epoch, ckpt_last.resolve())

            if not args.no_save_best:
                vm = record.get("valid")
                vl_best = vm.get("total") if isinstance(vm, dict) else None
                if (
                    vl_best is not None
                    and isinstance(vl_best, (int, float))
                    and math.isfinite(float(vl_best))
                    and float(vl_best) < best_valid_loss
                ):
                    best_valid_loss = float(vl_best)
                    save_training_checkpoint(
                        ckpt_best, model, optimizer, epoch, history, args, scheduler=scheduler
                    )
                    log.info("New best valid total loss=%.6f -> %s", best_valid_loss, ckpt_best.resolve())

            last_completed_epoch = epoch

    except KeyboardInterrupt:
        log.warning(
            "KeyboardInterrupt after last fully completed epoch %s — saving %s",
            last_completed_epoch,
            ckpt_last,
        )
        if args.checkpoint_every != 0 and last_completed_epoch > start_epoch:
            save_training_checkpoint(
                ckpt_last,
                model,
                optimizer,
                last_completed_epoch,
                history,
                args,
                scheduler=scheduler,
            )
        raise SystemExit(130) from None

    save_artifacts(args.output_dir, prepared, model, history, args)
    log.info("Saved artifacts to %s", args.output_dir.resolve())
    print(json.dumps({"status": "done", "output_dir": str(args.output_dir), "metadata": prepared.metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
