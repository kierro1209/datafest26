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
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[BOS]": 1,
    "[EOS]": 2,
    "[UNK]": 3,
}

GAP_BINS = ["START", "0D", "1_7D", "8_30D", "31_90D", "91_180D", "181_365D", "365PLUS", "UNKNOWN"]
SETTING_BINS = ["ED", "INPATIENT", "OUTPATIENT", "OBS", "OP_FACE", "NONE", "UNKNOWN"]
FACILITY_SIZE_BINS = ["RURAL_LT1000", "SMALL_1K_10K", "MID_10K_50K", "LARGE_50K_200K", "METRO_200K_PLUS", "UNKNOWN"]
TRANSFER_BINS = ["NO_TRANSFER", "LOCAL_TO_LARGER", "LARGER_TO_LOCAL", "SAME_FACILITY", "UNKNOWN"]


@dataclass
class PreparedData:
    sequences: list[dict[str, Any]]
    vocab: dict[str, int]
    gap_to_id: dict[str, int]
    setting_to_id: dict[str, int]
    dept_type_to_id: dict[str, int]
    facility_size_to_id: dict[str, int]
    region_to_id: dict[str, int]
    metadata: dict[str, Any]


def column_or_default(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index, dtype="object")


class PatientSequenceDataset(Dataset):
    def __init__(self, sequences: list[dict[str, Any]], max_seq_len: int) -> None:
        self.sequences = sequences
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        event_ids = seq["event_token_ids"][: self.max_seq_len]
        gap_ids = seq["gap_ids"][: self.max_seq_len]
        setting_ids = seq["setting_ids"][: self.max_seq_len]
        dept_ids = seq["dept_type_ids"][: self.max_seq_len]
        size_ids = seq["facility_size_ids"][: self.max_seq_len]
        region_ids = seq["region_ids"][: self.max_seq_len]

        input_ids = [SPECIAL_TOKENS["[BOS]"]] + event_ids[:-1]
        target_ids = event_ids
        attention_mask = [1] * len(event_ids)

        pad_n = self.max_seq_len - len(event_ids)
        if pad_n > 0:
            input_ids += [SPECIAL_TOKENS["[PAD]"]] * pad_n
            target_ids += [-100] * pad_n
            gap_ids += [-100] * pad_n
            setting_ids += [-100] * pad_n
            dept_ids += [-100] * pad_n
            size_ids += [-100] * pad_n
            region_ids += [-100] * pad_n
            attention_mask += [0] * pad_n

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target_event": torch.tensor(target_ids, dtype=torch.long),
            "target_gap": torch.tensor(gap_ids, dtype=torch.long),
            "target_setting": torch.tensor(setting_ids, dtype=torch.long),
            "target_dept_type": torch.tensor(dept_ids, dtype=torch.long),
            "target_facility_size": torch.tensor(size_ids, dtype=torch.long),
            "target_region": torch.tensor(region_ids, dtype=torch.long),
        }


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
        vocab_size: int,
        gap_size: int,
        setting_size: int,
        dept_type_size: int,
        facility_size_size: int,
        region_size: int,
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
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
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

        self.token_head = nn.Linear(d_model, vocab_size)
        self.gap_head = nn.Linear(d_model, gap_size)
        self.setting_head = nn.Linear(d_model, setting_size)
        self.dept_type_head = nn.Linear(d_model, dept_type_size)
        self.facility_size_head = nn.Linear(d_model, facility_size_size)
        self.region_head = nn.Linear(d_model, region_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.embedding(input_ids)
        x = self.positional(x)
        x = self.dropout(x)

        if self.backbone_name == "transformer":
            causal_mask = self._causal_mask(input_ids.size(1), input_ids.device)
            key_padding_mask = attention_mask == 0
            hidden = self.backbone(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        else:
            hidden, _ = self.backbone(x)

        return {
            "token": self.token_head(hidden),
            "gap": self.gap_head(hidden),
            "setting": self.setting_head(hidden),
            "dept_type": self.dept_type_head(hidden),
            "facility_size": self.facility_size_head(hidden),
            "region": self.region_head(hidden),
        }


def normalize_value(value: Any, max_len: int = 80) -> str:
    if value is None:
        return "UNKNOWN"
    try:
        if pd.isna(value):
            return "UNKNOWN"
    except Exception:
        pass
    s = str(value).strip()
    if not s or s.upper() in {"NA", "NAN", "NULL", "NONE"}:
        return "UNKNOWN"
    s = s.upper()
    s = "_".join(part for part in "".join(ch if ch.isalnum() else " " for ch in s).split())
    return s[:max_len] if s else "UNKNOWN"


def trueish(value: Any) -> bool:
    return normalize_value(value) in {"1", "TRUE", "T", "YES", "Y"}


def first_present(row: pd.Series, names: list[str], default: Any = "") -> Any:
    for name in names:
        if name in row.index:
            value = row.get(name)
            if value is not None and str(value).strip() != "" and str(value).strip().upper() != "NAN":
                return value
    return default


def safe_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2.0) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def facility_size_bin(population_value: Any) -> str:
    n = safe_float(population_value)
    if n is None:
        return "UNKNOWN"
    if n < 1000:
        return "RURAL_LT1000"
    if n < 10000:
        return "SMALL_1K_10K"
    if n < 50000:
        return "MID_10K_50K"
    if n < 200000:
        return "LARGE_50K_200K"
    return "METRO_200K_PLUS"


def gap_bin(days: Any, patient_event_index: Any) -> str:
    try:
        idx = int(patient_event_index)
    except Exception:
        idx = None
    if idx == 1:
        return "START"
    d = safe_float(days)
    if d is None:
        return "UNKNOWN"
    d = int(d)
    if d == 0:
        return "0D"
    if 1 <= d <= 7:
        return "1_7D"
    if 8 <= d <= 30:
        return "8_30D"
    if 31 <= d <= 90:
        return "31_90D"
    if 91 <= d <= 180:
        return "91_180D"
    if 181 <= d <= 365:
        return "181_365D"
    if d > 365:
        return "365PLUS"
    return "UNKNOWN"


def setting_bin(row: pd.Series) -> str:
    if trueish(row.get("IsEdVisit")):
        return "ED"
    if trueish(row.get("IsInpatientAdmission")) or trueish(row.get("IsHospitalAdmission")):
        return "INPATIENT"
    if trueish(row.get("IsObservation")):
        return "OBS"
    if trueish(row.get("IsOutpatientFaceToFaceVisit")):
        return "OP_FACE"
    if trueish(row.get("IsHospitalOutpatientVisit")):
        return "OUTPATIENT"
    return "NONE"


def distance_bin(patient_lat: Any, patient_lon: Any, dept_lat: Any, dept_lon: Any) -> str:
    lat1 = safe_float(patient_lat)
    lon1 = safe_float(patient_lon)
    lat2 = safe_float(dept_lat)
    lon2 = safe_float(dept_lon)
    if None in {lat1, lon1, lat2, lon2}:
        return "UNKNOWN"
    dist = haversine_km(lat1, lon1, lat2, lon2)
    if dist == 0:
        return "SAME_TRACT"
    if dist < 5:
        return "LT5KM"
    if dist < 20:
        return "5_20KM"
    if dist < 50:
        return "20_50KM"
    return "50PLUS"


def kmeans_regions(df: pd.DataFrame, num_clusters: int, random_seed: int) -> pd.DataFrame:
    work = df[["department_geo_lat", "department_geo_lon"]].dropna().drop_duplicates().copy()
    if work.empty:
        df["region_label"] = "UNKNOWN"
        return df
    points = work[["department_geo_lat", "department_geo_lon"]].to_numpy(dtype=np.float64)
    k = max(1, min(num_clusters, len(points)))
    rng = np.random.default_rng(random_seed)
    centroids = points[rng.choice(len(points), size=k, replace=False)]
    for _ in range(25):
        distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assignment = distances.argmin(axis=1)
        new_centroids = []
        for cluster_idx in range(k):
            members = points[assignment == cluster_idx]
            if len(members) == 0:
                new_centroids.append(centroids[cluster_idx])
            else:
                new_centroids.append(members.mean(axis=0))
        new_centroids = np.vstack(new_centroids)
        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
            break
        centroids = new_centroids
    distances = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    assignment = distances.argmin(axis=1)
    work["region_label"] = [f"REGION_{idx}" for idx in assignment]
    return df.merge(work, on=["department_geo_lat", "department_geo_lon"], how="left")


def maybe_join_geography(event_df: pd.DataFrame, census_path: Path) -> pd.DataFrame:
    if not census_path.exists():
        event_df["patient_geo_lat"] = np.nan
        event_df["patient_geo_lon"] = np.nan
        event_df["department_geo_lat"] = np.nan
        event_df["department_geo_lon"] = np.nan
        event_df["department_geo_population"] = np.nan
        return event_df

    geo = pd.read_csv(census_path, dtype=str, usecols=["GEOID", "PopulationValue", "CENTLAT", "CENTLON"])
    patient_geo = geo.rename(columns={
        "GEOID": "patient_geoid",
        "PopulationValue": "patient_geo_population",
        "CENTLAT": "patient_geo_lat",
        "CENTLON": "patient_geo_lon",
    })
    dept_geo = geo.rename(columns={
        "GEOID": "department_geoid",
        "PopulationValue": "department_geo_population",
        "CENTLAT": "department_geo_lat",
        "CENTLON": "department_geo_lon",
    })

    if "CensusBlockGroupFipsCode" in event_df.columns:
        event_df = event_df.merge(patient_geo, left_on="CensusBlockGroupFipsCode", right_on="patient_geoid", how="left")
    else:
        event_df["patient_geo_population"] = np.nan
        event_df["patient_geo_lat"] = np.nan
        event_df["patient_geo_lon"] = np.nan

    dept_key = None
    for candidate in ["department_CensusTract", "CensusTract"]:
        if candidate in event_df.columns:
            dept_key = candidate
            break
    if dept_key is not None:
        event_df = event_df.merge(dept_geo, left_on=dept_key, right_on="department_geoid", how="left")
    else:
        event_df["department_geo_population"] = np.nan
        event_df["department_geo_lat"] = np.nan
        event_df["department_geo_lon"] = np.nan

    return event_df


def compute_transfer_labels(df: pd.DataFrame) -> pd.Series:
    prev_signature = df.groupby("PatientDurableKey")["facility_signature"].shift(1)
    prev_size = df.groupby("PatientDurableKey")["facility_size_bin"].shift(1)

    size_rank = {name: idx for idx, name in enumerate(FACILITY_SIZE_BINS)}
    labels: list[str] = []
    for idx, row in df.iterrows():
        prev_sig = prev_signature.loc[idx]
        prev_size_val = prev_size.loc[idx]
        curr_sig = row["facility_signature"]
        curr_size_val = row["facility_size_bin"]
        if pd.isna(prev_sig) or prev_size_val is None or prev_size_val == "UNKNOWN" or curr_size_val == "UNKNOWN":
            labels.append("UNKNOWN" if int(row["patient_event_index"]) > 1 else "NO_TRANSFER")
            continue
        if prev_sig == curr_sig:
            labels.append("SAME_FACILITY")
            continue
        prev_rank = size_rank.get(prev_size_val)
        curr_rank = size_rank.get(curr_size_val)
        if prev_rank is None or curr_rank is None:
            labels.append("UNKNOWN")
        elif curr_rank > prev_rank:
            labels.append("LOCAL_TO_LARGER")
        elif curr_rank < prev_rank:
            labels.append("LARGER_TO_LOCAL")
        else:
            labels.append("NO_TRANSFER")
    return pd.Series(labels, index=df.index)


def composite_token(row: pd.Series) -> str:
    parts = [
        f"PAT_AGEBIN_{normalize_value(row.get('PatientBirthYearBin'))}",
        f"PAT_SEX_{normalize_value(row.get('SexAssignedAtBirth'))}",
        f"PAT_RACE_{normalize_value(row.get('OmbRace'))}",
        f"PAT_ETH_{normalize_value(row.get('OmbEthnicity'))}",
        f"PAT_SMOKE_{normalize_value(row.get('SmokingStatus'))}",
        f"PAT_MARITAL_{normalize_value(row.get('MaritalStatus'))}",
        f"PAT_MYCHART_{'YES' if trueish(row.get('MyChartStatus')) else 'NO'}",
        f"PAT_POPBIN_{facility_size_bin(row.get('patient_geo_population'))}",
        f"PAT_GEO_KNOWN_{'YES' if normalize_value(row.get('CensusBlockGroupFipsCode')) != 'UNKNOWN' else 'NO'}",
        f"PAT_SDOH_OBS_{'YES' if trueish(row.get('sdoh_any_observed')) else 'NO'}",
        f"PAT_SDOH_DOMAINCOUNT_{domain_count_bin(row.get('sdoh_num_domains_answered'))}",
        f"PAT_SDOH_DOMAIN_{domain_token(row.get('sdoh_domains_observed'))}",
        f"EVT_TYPE_{normalize_value(first_present(row, ['event_type', 'Type']))}",
        f"EVT_DXG_{normalize_value(row.get('GroupCode'))}",
        f"EVT_DX_{normalize_value(row.get('DiagnosisValue'))}",
        f"EVT_SETTING_{row.get('setting_bin', 'UNKNOWN')}",
        f"EVT_DEPT_TYPE_{normalize_value(row.get('DepartmentType'))}",
        f"EVT_DEPT_SPEC_{normalize_value(row.get('DepartmentSpecialty'))}",
        f"EVT_FACILITY_SIZE_{row.get('facility_size_bin', 'UNKNOWN')}",
        f"EVT_DIST_{row.get('distance_bin', 'UNKNOWN')}",
        f"EVT_REGION_{row.get('region_label', 'UNKNOWN')}",
        f"EVT_TRANSFER_{row.get('transfer_bin', 'UNKNOWN')}",
        f"EVT_GAP_{row.get('gap_bin', 'UNKNOWN')}",
    ]
    seen: set[str] = set()
    deduped: list[str] = []
    for part in parts:
        if part not in seen:
            deduped.append(part)
            seen.add(part)
    return "EVENT_COMPOSITE::" + "|".join(deduped)


def domain_count_bin(value: Any) -> str:
    n = safe_float(value)
    if n is None or n <= 0:
        return "0"
    if int(n) == 1:
        return "1"
    if int(n) == 2:
        return "2"
    return "3PLUS"


def domain_token(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return "NONE"
    parts = [normalize_value(part) for part in str(value).replace("|", ",").split(",")]
    parts = [part for part in parts if part != "UNKNOWN"]
    if not parts:
        return "NONE"
    return "__".join(sorted(set(parts)))


def load_and_prepare_dataframe(input_path: Path, census_path: Path, num_regions: int, random_seed: int) -> pd.DataFrame:
    df = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    df = maybe_join_geography(df, census_path)

    sort_cols = [
        "event_PatientDurableKey",
        "event_date",
        "event_time",
        "event_EncounterKey",
        "event_index_within_encounter",
        "event_id",
    ]
    for col in sort_cols:
        if col not in df.columns:
            df[col] = ""

    df["PatientDurableKey"] = column_or_default(df, "event_PatientDurableKey")
    patient_fallback = column_or_default(df, "PatientDurableKey")
    df["PatientDurableKey"] = df["PatientDurableKey"].where(df["PatientDurableKey"].astype(str).str.strip() != "", patient_fallback)
    df["EncounterKey"] = column_or_default(df, "event_EncounterKey")
    encounter_fallback = column_or_default(df, "EncounterKey")
    df["EncounterKey"] = df["EncounterKey"].where(df["EncounterKey"].astype(str).str.strip() != "", encounter_fallback)
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    df["patient_event_index"] = df.groupby("PatientDurableKey").cumcount() + 1
    df["previous_event_date"] = df.groupby("PatientDurableKey")["event_date"].shift(1)
    event_dates = pd.to_datetime(df["event_date"], errors="coerce")
    prev_dates = pd.to_datetime(df["previous_event_date"], errors="coerce")
    df["days_since_previous_event"] = (event_dates - prev_dates).dt.days
    df["gap_bin"] = [gap_bin(days, idx) for days, idx in zip(df["days_since_previous_event"], df["patient_event_index"], strict=False)]
    df["setting_bin"] = df.apply(setting_bin, axis=1)
    df["facility_size_bin"] = df["department_geo_population"].map(facility_size_bin)
    df["distance_bin"] = [
        distance_bin(p_lat, p_lon, d_lat, d_lon)
        for p_lat, p_lon, d_lat, d_lon in zip(
            df.get("patient_geo_lat", pd.Series(index=df.index, dtype=object)),
            df.get("patient_geo_lon", pd.Series(index=df.index, dtype=object)),
            df.get("department_geo_lat", pd.Series(index=df.index, dtype=object)),
            df.get("department_geo_lon", pd.Series(index=df.index, dtype=object)),
            strict=False,
        )
    ]
    df = kmeans_regions(df, num_clusters=num_regions, random_seed=random_seed)
    df["region_label"] = df["region_label"].fillna("UNKNOWN")
    dept_tract = column_or_default(df, "department_CensusTract")
    dept_tract = dept_tract.where(dept_tract.astype(str).str.strip() != "", column_or_default(df, "CensusTract"))
    df["facility_signature"] = (
        column_or_default(df, "DepartmentType").astype(str)
        + "|" + column_or_default(df, "DepartmentSpecialty").astype(str)
        + "|" + dept_tract.astype(str)
    )
    df["transfer_bin"] = compute_transfer_labels(df)
    df["event_composite_token"] = df.apply(composite_token, axis=1)
    return df


def build_mapping(values: list[str], base_values: list[str] | None = None) -> dict[str, int]:
    ordered: list[str] = []
    if base_values is not None:
        ordered.extend(base_values)
    for value in sorted(set(values)):
        if value not in ordered:
            ordered.append(value)
    return {value: idx for idx, value in enumerate(ordered)}


def prepare_sequences(df: pd.DataFrame, min_token_freq: int) -> PreparedData:
    counts = df["event_composite_token"].value_counts()
    vocab = dict(SPECIAL_TOKENS)
    next_id = max(vocab.values()) + 1
    for token, freq in counts.items():
        if int(freq) >= min_token_freq:
            vocab[token] = next_id
            next_id += 1

    df["event_token_id"] = df["event_composite_token"].map(lambda x: vocab.get(x, SPECIAL_TOKENS["[UNK]"]))
    gap_to_id = build_mapping(df["gap_bin"].fillna("UNKNOWN").astype(str).tolist(), GAP_BINS)
    setting_to_id = build_mapping(df["setting_bin"].fillna("UNKNOWN").astype(str).tolist(), SETTING_BINS)
    dept_type_to_id = build_mapping(column_or_default(df, "DepartmentType", "UNKNOWN").map(normalize_value).tolist(), ["UNKNOWN"])
    facility_size_to_id = build_mapping(df["facility_size_bin"].fillna("UNKNOWN").astype(str).tolist(), FACILITY_SIZE_BINS)
    region_to_id = build_mapping(df["region_label"].fillna("UNKNOWN").astype(str).tolist(), ["UNKNOWN"])

    sequences: list[dict[str, Any]] = []
    for patient_id, group in df.groupby("PatientDurableKey", sort=False):
        group = group.sort_values("patient_event_index")
        event_token_ids = group["event_token_id"].astype(int).tolist()
        if len(event_token_ids) < 2:
            continue
        sequences.append({
            "patient_id": patient_id,
            "event_token_ids": event_token_ids,
            "gap_ids": [gap_to_id.get(v, gap_to_id["UNKNOWN"]) for v in group["gap_bin"].fillna("UNKNOWN").astype(str)],
            "setting_ids": [setting_to_id.get(v, setting_to_id.get("UNKNOWN", 0)) for v in group["setting_bin"].fillna("UNKNOWN").astype(str)],
            "dept_type_ids": [dept_type_to_id.get(normalize_value(v), dept_type_to_id["UNKNOWN"]) for v in column_or_default(group, "DepartmentType", "UNKNOWN").tolist()],
            "facility_size_ids": [facility_size_to_id.get(v, facility_size_to_id["UNKNOWN"]) for v in group["facility_size_bin"].fillna("UNKNOWN").astype(str)],
            "region_ids": [region_to_id.get(v, region_to_id["UNKNOWN"]) for v in group["region_label"].fillna("UNKNOWN").astype(str)],
        })

    metadata = {
        "n_events": int(len(df)),
        "n_patients": int(df["PatientDurableKey"].nunique()),
        "n_training_sequences": int(len(sequences)),
        "vocab_size": int(len(vocab)),
        "gap_classes": int(len(gap_to_id)),
        "setting_classes": int(len(setting_to_id)),
        "dept_type_classes": int(len(dept_type_to_id)),
        "facility_size_classes": int(len(facility_size_to_id)),
        "region_classes": int(len(region_to_id)),
    }
    return PreparedData(
        sequences=sequences,
        vocab=vocab,
        gap_to_id=gap_to_id,
        setting_to_id=setting_to_id,
        dept_type_to_id=dept_type_to_id,
        facility_size_to_id=facility_size_to_id,
        region_to_id=region_to_id,
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
    token_loss = ce(outputs["token"].transpose(1, 2), batch["target_event"])
    gap_loss = ce(outputs["gap"].transpose(1, 2), batch["target_gap"])
    setting_loss = ce(outputs["setting"].transpose(1, 2), batch["target_setting"])
    dept_loss = ce(outputs["dept_type"].transpose(1, 2), batch["target_dept_type"])
    size_loss = ce(outputs["facility_size"].transpose(1, 2), batch["target_facility_size"])
    region_loss = ce(outputs["region"].transpose(1, 2), batch["target_region"])
    total = (
        token_loss
        + weights["gap"] * gap_loss
        + weights["setting"] * setting_loss
        + weights["dept_type"] * dept_loss
        + weights["facility_size"] * size_loss
        + weights["region"] * region_loss
    )
    metrics = {
        "token": float(token_loss.detach().cpu()),
        "gap": float(gap_loss.detach().cpu()),
        "setting": float(setting_loss.detach().cpu()),
        "dept_type": float(dept_loss.detach().cpu()),
        "facility_size": float(size_loss.detach().cpu()),
        "region": float(region_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, metrics


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device, weights: dict[str, float]) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {"token": 0.0, "gap": 0.0, "setting": 0.0, "dept_type": 0.0, "facility_size": 0.0, "region": 0.0, "total": 0.0}
    batches = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train_mode):
            outputs = model(batch["input_ids"], batch["attention_mask"])
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
        "vocab": prepared.vocab,
        "gap_to_id": prepared.gap_to_id,
        "setting_to_id": prepared.setting_to_id,
        "dept_type_to_id": prepared.dept_type_to_id,
        "facility_size_to_id": prepared.facility_size_to_id,
        "region_to_id": prepared.region_to_id,
        "training_history": history,
    }
    (output_dir / "patient_event_model_artifacts.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a multi-task autoregressive patient event sequence model.")
    p.add_argument("--input", type=Path, default=Path("data/processed/event_enriched.csv.gz"))
    p.add_argument("--census", type=Path, default=Path("data/raw/tigercensuscodes.csv"))
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
    p.add_argument("--w-size", type=float, default=1.0)
    p.add_argument("--w-region", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = load_and_prepare_dataframe(args.input, args.census, args.num_regions, args.seed)
    prepared = prepare_sequences(df, min_token_freq=args.min_token_freq)
    if not prepared.sequences:
        raise SystemExit("No patient sequences with at least two events were found.")

    train_sequences, valid_sequences = split_sequences(prepared.sequences, args.valid_fraction, args.seed)
    if not train_sequences:
        train_sequences = prepared.sequences
        valid_sequences = []

    train_ds = PatientSequenceDataset(train_sequences, max_seq_len=args.max_seq_len)
    valid_ds = PatientSequenceDataset(valid_sequences, max_seq_len=args.max_seq_len) if valid_sequences else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False) if valid_ds is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatientEventSequenceModel(
        vocab_size=len(prepared.vocab),
        gap_size=len(prepared.gap_to_id),
        setting_size=len(prepared.setting_to_id),
        dept_type_size=len(prepared.dept_type_to_id),
        facility_size_size=len(prepared.facility_size_to_id),
        region_size=len(prepared.region_to_id),
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
        "gap": args.w_gap,
        "setting": args.w_setting,
        "dept_type": args.w_dept,
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
