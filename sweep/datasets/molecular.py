"""Molecular biology datasets: PDB (biotite), ProteinNet, ESM Atlas.

Working offline unless a structure is fetched from RCSB for the first time.
`protein_stats` returns a compact structural report for a structure in the pdb
registry or an RCSB PDB id.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .registry import dataset_dir

AMINO_MASS = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G": 75.07, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}


def load_molecular(name: str, split: Optional[str] = None, limit: Optional[int] = None) -> dict[str, Any]:
    if name == "pdb":
        path = dataset_dir() / "pdb"
        path.mkdir(parents=True, exist_ok=True)
        return {"rows": [], "row_count": 0, "columns": [],
                "path": str(path), "notes": "Structures fetched on demand via biotite from rcsb.org. Say 'data protein <PDB_ID>'."}
    if name == "proteinnet":
        path = dataset_dir() / "proteinnet"
        path.mkdir(parents=True, exist_ok=True)
        return {"rows": [], "row_count": 0, "columns": [],
                "path": str(path),
                "notes": "ProteinNet files are parsed locally with proteinnetpy. Put a .pnet/.txt file in the folder and say 'data protein <file path>'."}
    if name == "esm_atlas":
        return {"rows": [], "row_count": 0, "columns": [],
                "path": str(dataset_dir() / "esm_atlas"),
                "notes": "ESM (fair-esm) embeddings are computed on demand for a sequence. Say 'data esm AAACGT...'."
                         " The first call downloads the ~8M model weights (one time)."}
    raise ValueError(f"Unknown molecular dataset: {name}")


def protein_stats(target: str) -> dict[str, Any]:
    """Structural report for a local ProteinNet file, cached structure, or PDB id."""
    candidate = Path(target).expanduser()
    if candidate.exists():
        return _proteinnet_report(candidate)
    lowered = target.strip().lower()
    if lowered.startswith(("http://", "https://")):
        raise RuntimeError("Give me a PDB id (e.g. '7tim') or a local structure file path.")
    pdb_id = lowered.replace("-", "").upper() if len(lowered.replace("-", "")) == 4 else lowered.upper()
    return _pdb_report(pdb_id)


def _pdb_report(pdb_id: str) -> dict[str, Any]:
    try:
        import biotite.database.rcsb as rcsb
        import biotite.structure.io.pdb as pdbio

        target_dir = dataset_dir() / "pdb"
        target_dir.mkdir(parents=True, exist_ok=True)
        path = rcsb.fetch(pdb_id, format="pdb", target_dir=target_dir)
        pdb_file = pdbio.PDBFile.read(path)
        structure = pdb_file.get_structure(model=1)
        seq = _sequence_from_structure(structure)
        composition = _composition(seq)
        chains = sorted({str(a) for a in structure.chain_id})
        return {
            "pdb_id": pdb_id,
            "description": f"Structure fetched from PDB ({path})",
            "chains": chains,
            "sequence_length": len(seq),
            "sequence": seq[:240],
            "composition": composition,
            "molecular_weight_approx": round(_mw(seq), 1),
            "residue_count": int(structure.array_length()),
        }
    except Exception as exc:  # noqa: BLE001 — surface a friendly message
        raise RuntimeError(f"Couldn't fetch PDB {pdb_id!r}: {exc}") from exc


def _sequence_from_structure(structure) -> str:
    aa = structure.res_name
    three = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
        "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V",
    }
    seen = set()
    seq = ""
    for i, name in enumerate(aa):
        one = three.get(str(name), "X")
        key = (i, name)
        if key in seen or one == "X":
            continue
        seen.add(key)
        seq += one
    return seq[:400]


def _proteinnet_report(path: Path) -> dict[str, Any]:
    try:
        from proteinnet import parse_file  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"proteinnetpy can't parse this file on your machine ({exc}).") from exc
    rows = list(parse_file(str(path)))
    if not rows:
        raise RuntimeError(f"No records parsed from {path}.")
    sample = rows[0]
    primary = ""
    for field, value in sample:  # (field_name, value) pairs from proteinnetpy
        if field == "primary":
            primary = str(value)
    return {
        "file": str(path),
        "records": len(rows),
        "sequence_length": len(primary),
        "sequence": primary[:240],
        "composition": _composition(primary),
        "molecular_weight_approx": round(_mw(primary), 1),
    }


def _composition(seq: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in seq.upper():
        if ch in AMINO_MASS:
            counts[ch] = counts.get(ch, 0) + 1
    return counts


def _mw(seq: str) -> float:
    return sum(AMINO_MASS.get(ch, 128.0) for ch in seq.upper())


def esm_embed(sequence: str) -> dict[str, Any]:
    """Embed a protein sequence with a local ESM model (downloads weights once)."""
    try:
        import esm  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"fair-esm isn't importable: {exc}") from exc
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    import torch

    with torch.no_grad():
        labels, strs, toks = batch_converter([("seq", str(sequence).upper())])
        result = model(toks, repr_layers=[6], return_contacts=False)
    token_embeddings = result["representations"][6]
    pooled = token_embeddings[0, 1:-1].mean(dim=0)
    return {
        "model": "esm2_t6_8M_UR50D",
        "embedding_dim": int(pooled.shape[0]),
        "pooled_norm": float(pooled.norm().item()),
        "tokens": int(toks.shape[1]),
    }


def esm_status() -> Optional[str]:
    path = dataset_dir() / "esm_atlas" / "weights"
    return "weights present" if path.exists() else "weights not yet downloaded"