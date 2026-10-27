"""
00_probe_eval_set.py

Builds the probe evaluation CSV consumed by 09_mutual_information.py (paper Fig 7d, S9,
Section C: 7,484 molecules, 3,742 R / 3,742 S).

Source: a CSV of training pairs with columns ``canonical`` and ``random`` whose first column
is a molecule id, made of n_buckets equal consecutive length buckets (paper:
sampled_100k.csv, 5 x 20,000 rows).

Filtering and labels:
  - canonical SMILES parsed by RDKit; molecules with exactly one chiral centre, which is
    assigned (FindMolChiralCenters with includeUnassigned=True has length 1 and the centre
    has a CIP label) are kept; R -> rs_label_id 0, S -> 1
  - canonical_/random_chiral_token: '@@' if the string has exactly one '@@' and no other '@',
    '@' if it has exactly one '@' and no '@@', else none; ids '@' -> 0, '@@' -> 1, none -> -1
  - balanced sampling: min(n_R, n_S, max_per_class) per class with
    DataFrame.sample(random_state=seed), then shuffled with sample(frac=1, random_state=seed)

Outputs in output_dir:
  single_chiral_rs_probe_{N}.csv          (N = "{k}k" for multiples of 1,000, else the count)
  single_chiral_rs_probe_{N}_summary.json

Config YAML:
  source_csv: /path/to/sampled_100k.csv
  output_dir
  seed (42), max_per_class (5000), n_buckets (5)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def extract_chiral_token(smi: str) -> str | None:
    """'@@' or '@' if the SMILES holds exactly one chirality mark of one kind, else None."""
    smi_masked = smi.replace("@@", "\x00\x00")
    n_double = smi.count("@@")
    n_single = smi_masked.count("@")
    if n_double == 1 and n_single == 0:
        return "@@"
    if n_double == 0 and n_single == 1:
        return "@"
    return None


def chiral_token_id(token: str | None) -> int:
    return {"@": 0, "@@": 1}.get(token, -1)


def build_probe_set(df: pd.DataFrame, seed: int, max_per_class: int, n_buckets: int) -> tuple[pd.DataFrame, dict]:
    """Filter, label and balance the source pairs. ``df`` is indexed by molecule id."""
    if len(df) % n_buckets != 0:
        raise ValueError(f"{len(df)} source rows cannot be split into {n_buckets} equal buckets")
    bucket_size = len(df) // n_buckets
    df = df.reset_index(drop=False)
    df.rename(columns={df.columns[0]: "mol_id"}, inplace=True)
    df["bucket"] = np.clip(df.index.to_numpy() // bucket_size + 1, 1, n_buckets)

    records = []
    counts = {"invalid": 0, "achiral": 0, "multi_chiral": 0, "unassigned": 0}
    for row in df.itertuples():
        canonical = str(row.canonical)
        random_smiles = str(row.random)
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            counts["invalid"] += 1
            continue
        centers_all = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
        n_unassigned_centers = len(centers_all) - len(centers)
        if len(centers_all) == 0:
            counts["achiral"] += 1
            continue
        if len(centers_all) != 1 or n_unassigned_centers > 0:
            if n_unassigned_centers > 0 and len(centers) == 0:
                counts["unassigned"] += 1
            else:
                counts["multi_chiral"] += 1
            continue
        atom_idx, rs = centers[0]
        can_tok = extract_chiral_token(canonical)
        rnd_tok = extract_chiral_token(random_smiles)
        records.append({
            "mol_id": row.mol_id,
            "bucket": int(row.bucket),
            "canonical": canonical,
            "random": random_smiles,
            "rs_label": "R" if rs == "R" else "S",
            "rs_label_id": 0 if rs == "R" else 1,
            "chiral_center_atom_idx": int(atom_idx),
            "canonical_chiral_token": can_tok,
            "random_chiral_token": rnd_tok,
            "canonical_chiral_token_id": chiral_token_id(can_tok),
            "random_chiral_token_id": chiral_token_id(rnd_tok),
            "canonical_length": len(canonical),
            "random_length": len(random_smiles),
        })
    valid_df = pd.DataFrame(records)

    df_r = valid_df[valid_df["rs_label_id"] == 0]
    df_s = valid_df[valid_df["rs_label_id"] == 1]
    n_per_class = min(len(df_r), len(df_s), max_per_class)
    final_df = pd.concat([
        df_r.sample(n=n_per_class, random_state=seed),
        df_s.sample(n=n_per_class, random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    buckets = {}
    for b in range(1, n_buckets + 1):
        valid_b = valid_df[valid_df["bucket"] == b]
        used_b = final_df[final_df["bucket"] == b]
        buckets[str(b)] = {
            "n_total_in_bucket": bucket_size,
            "n_valid_R": int((valid_b["rs_label_id"] == 0).sum()),
            "n_valid_S": int((valid_b["rs_label_id"] == 1).sum()),
            "n_used_R": int((used_b["rs_label_id"] == 0).sum()),
            "n_used_S": int((used_b["rs_label_id"] == 1).sum()),
        }
    n_total = len(final_df)
    summary = {
        "seed": seed,
        "max_per_class": max_per_class,
        "n_source_rows": len(df),
        **{f"n_{key}_removed": value for key, value in counts.items()},
        "n_valid_extracted": len(valid_df),
        "n_R_available": len(df_r),
        "n_S_available": len(df_s),
        "n_total": n_total,
        "n_R": int((final_df["rs_label_id"] == 0).sum()),
        "n_S": int((final_df["rs_label_id"] == 1).sum()),
        "n_canonical_chiral_token_none": int(final_df["canonical_chiral_token"].isna().sum()),
        "n_random_chiral_token_none": int(final_df["random_chiral_token"].isna().sum()),
        "buckets": buckets,
    }
    return final_df, summary


def main():
    parser = argparse.ArgumentParser(description="Build the single-chiral-centre R/S probe CSV.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    source = Path(cfg["source_csv"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source, index_col=0)
    for col in ("canonical", "random"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {source}")
    final_df, summary = build_probe_set(
        df, int(cfg.get("seed", 42)), int(cfg.get("max_per_class", 5000)), int(cfg.get("n_buckets", 5))
    )
    summary["source_csv"] = str(source)

    n_total = len(final_df)
    label = f"{n_total // 1000}k" if n_total % 1000 == 0 else str(n_total)
    csv_path = output_dir / f"single_chiral_rs_probe_{label}.csv"
    final_df.to_csv(csv_path, index=False)
    summary["output_csv"] = str(csv_path)
    with open(output_dir / f"single_chiral_rs_probe_{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {csv_path} ({n_total} rows: R={summary['n_R']}, S={summary['n_S']})")


if __name__ == "__main__":
    main()
