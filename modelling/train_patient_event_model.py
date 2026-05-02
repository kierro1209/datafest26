#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

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


class PatientSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[dict[str, Any]],
        max_seq_len: int,
        sdoh_fields: list[str],
        patient_context_fields: list[str],
        patient_numeric_fields: list[str],
    ) -> None:
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.sdoh_fields = sdoh_fields
        self.patient_context_fields = patient_context_fields
        self.patient_numeric_fields = patient_numeric_fields

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
        for field in self.sdoh_fields:
            batch[f"input_{field}"] = torch.tensor(sdoh_inputs[field], dtype=torch.long)
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
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.pad_token_id = pad_token_id
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
        x = self.positional(x)
        x = self.dropout(x)

        attention_mask = batch["attention_mask"]
        if self.backbone_name == "transformer":
            causal_mask = self._causal_mask(x.size(1), x.device)
            key_padding_mask = attention_mask == 0
            hidden = self.backbone(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        else:
            hidden, _ = self.backbone(x)

        return {
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
    if name.startswith("patient_sequences_") and name.endswith(".pt"):
        suffix = name[len("patient_sequences_") : -len(".pt")]
        return input_path.with_name(f"sequence_model_vocab_{suffix}.json")
    return input_path.with_suffix(".json")


def _load_vocab_payload(vocab_path: Path) -> dict[str, Any]:
    if not vocab_path.exists():
        return {}
    return json.loads(vocab_path.read_text(encoding="utf-8"))


def _first_available(payloads: list[dict[str, Any]], key: str, default: Any = None) -> Any:
    for payload in payloads:
        if isinstance(payload, dict) and key in payload and payload[key] is not None:
            return payload[key]
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
    numeric_candidates = ["patient_lat", "patient_lon", "patient_population"]
    patient_numeric_fields = [
        key for key in numeric_candidates if key in (sample.get("patient_context_values") or {})
    ]

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


def compute_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    type_loss = ce(outputs["type"].transpose(1, 2), batch["target_type"])
    event_description_loss = ce(outputs["event_description"].transpose(1, 2), batch["target_event_description"])
    group_code_loss = ce(outputs["group_code"].transpose(1, 2), batch["target_group_code"])
    diagnosis_value_loss = ce(outputs["diagnosis_value"].transpose(1, 2), batch["target_diagnosis_value"])
    gap_loss = ce(outputs["gap"].transpose(1, 2), batch["target_gap"])
    setting_loss = ce(outputs["setting"].transpose(1, 2), batch["target_setting"])
    dept_loss = ce(outputs["dept_type"].transpose(1, 2), batch["target_dept_type"])
    dept_specialty_loss = ce(outputs["dept_specialty"].transpose(1, 2), batch["target_dept_specialty"])
    size_loss = ce(outputs["facility_size"].transpose(1, 2), batch["target_facility_size"])
    region_loss = ce(outputs["region"].transpose(1, 2), batch["target_region"])
    total = (
        weights["type"] * type_loss
        + weights["event_description"] * event_description_loss
        + weights["group_code"] * group_code_loss
        + weights["diagnosis_value"] * diagnosis_value_loss
        + weights["gap"] * gap_loss
        + weights["setting"] * setting_loss
        + weights["dept_type"] * dept_loss
        + weights["dept_specialty"] * dept_specialty_loss
        + weights["facility_size"] * size_loss
        + weights["region"] * region_loss
    )
    metrics = {
        "type": float(type_loss.detach().cpu()),
        "event_description": float(event_description_loss.detach().cpu()),
        "group_code": float(group_code_loss.detach().cpu()),
        "diagnosis_value": float(diagnosis_value_loss.detach().cpu()),
        "gap": float(gap_loss.detach().cpu()),
        "setting": float(setting_loss.detach().cpu()),
        "dept_type": float(dept_loss.detach().cpu()),
        "dept_specialty": float(dept_specialty_loss.detach().cpu()),
        "facility_size": float(size_loss.detach().cpu()),
        "region": float(region_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, metrics


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device, weights: dict[str, float]) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {
        "type": 0.0,
        "event_description": 0.0,
        "group_code": 0.0,
        "diagnosis_value": 0.0,
        "gap": 0.0,
        "setting": 0.0,
        "dept_type": 0.0,
        "dept_specialty": 0.0,
        "facility_size": 0.0,
        "region": 0.0,
        "total": 0.0,
    }
    batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train_mode):
            outputs = model(batch)
            loss, metrics = compute_losses(outputs, batch, weights)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        for key, value in metrics.items():
            totals[key] += value
        batches += 1
    if batches == 0:
        return totals
    return {key: value / batches for key, value in totals.items()}


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
        "training_history": history,
    }
    (output_dir / "patient_event_model_artifacts.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a multi-task autoregressive patient event sequence model.")
    p.add_argument("--input", type=Path, default=Path("modelling/token_sequence_model/patient_sequences_encounter_only_with_sdoh_status_and_fips.pt"))
    p.add_argument("--vocab-json", type=Path, default=Path("modelling/token_sequence_model/sequence_model_vocab_encounter_only_with_sdoh_status_and_fips.json"))
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
    p.add_argument("--valid-fraction", type=float, default=0.1)
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
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
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
        sdoh_fields=[f"sdoh_{idx}" for idx in range(len(prepared.sdoh_fields))],
        patient_context_fields=prepared.patient_context_fields,
        patient_numeric_fields=prepared.patient_numeric_fields,
    )
    valid_ds = PatientSequenceDataset(
        indexed_valid_sequences,
        max_seq_len=args.max_seq_len,
        sdoh_fields=[f"sdoh_{idx}" for idx in range(len(prepared.sdoh_fields))],
        patient_context_fields=prepared.patient_context_fields,
        patient_numeric_fields=prepared.patient_numeric_fields,
    ) if indexed_valid_sequences else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False) if valid_ds is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    ).to(device)

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

    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, weights)
        valid_metrics = run_epoch(model, valid_loader, None, device, weights) if valid_loader is not None else None
        record = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
        history.append(record)
        print(json.dumps(record))

    save_artifacts(args.output_dir, prepared, model, history, args)
    print(json.dumps({"status": "done", "output_dir": str(args.output_dir), "metadata": prepared.metadata}, indent=2))


if __name__ == "__main__":
    main()
