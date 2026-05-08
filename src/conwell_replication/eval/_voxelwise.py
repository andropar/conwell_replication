"""Sidecar storage for voxel-wise SRPR scores."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


class VoxelwiseScoreWriter:
    """Collect voxel-wise SRPR vectors and write one sidecar per eval shard."""

    def __init__(self, out_dir: Path, stem: str, enabled: bool):
        self.enabled = enabled
        self.out_dir = out_dir
        self.stem = stem
        self.rows: List[dict] = []
        self.scores: List[np.ndarray] = []

    @property
    def score_path(self) -> Path:
        return self.out_dir / f"{self.stem}.scores.npy"

    @property
    def index_path(self) -> Path:
        return self.out_dir / f"{self.stem}.index.parquet"

    def add(self, score_vector: np.ndarray, metadata: dict) -> float:
        """Store one voxel-wise vector and return its scalar mean score."""
        score_vector = np.asarray(score_vector, dtype=np.float32)
        score = float(score_vector.mean(dtype=np.float64))
        if self.enabled:
            row = dict(metadata)
            row["score"] = score
            row["voxelwise_row"] = len(self.scores)
            row["n_voxels"] = int(score_vector.shape[0])
            self.rows.append(row)
            self.scores.append(score_vector)
        return score

    def write(self) -> Optional[tuple[Path, Path]]:
        """Write sidecars atomically enough for skip-existing checks."""
        if not self.enabled or not self.scores:
            return None

        self.out_dir.mkdir(parents=True, exist_ok=True)
        scores = np.stack(self.scores, axis=0).astype(np.float32, copy=False)
        index = pd.DataFrame(self.rows)

        tmp_scores = self.score_path.with_name(self.score_path.name + ".tmp")
        tmp_index = self.index_path.with_name(self.index_path.name + ".tmp")
        with tmp_scores.open("wb") as f:
            np.save(f, scores)
        index.to_parquet(tmp_index, index=False)
        tmp_scores.replace(self.score_path)
        tmp_index.replace(self.index_path)
        return self.score_path, self.index_path
