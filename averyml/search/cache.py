"""Synthesis output caching for grid search.

When running a grid search over (T_train, T_eval), cells that share
the same T_train can reuse synthesis outputs instead of re-sampling.
This saves 40-80% of wall-clock time on typical grids.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from averyml.config.synthesis import SynthesisConfig
from averyml.synthesis.sampler import compute_cache_key

logger = logging.getLogger(__name__)


class SynthesisCache:
    """Content-addressed cache for synthesis outputs.

    Cache key is derived from: model_id, prompt_source, prompt_dataset,
    temperature, top_k, top_p, min_p, max_tokens, seed, n_samples.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, config: SynthesisConfig) -> Path | None:
        """Check if synthesis output exists in cache. Returns path or None."""
        key = compute_cache_key(config)
        cached_path = self.cache_dir / f"synth_{key}.jsonl"
        meta_path = self.cache_dir / f"synth_{key}.meta.json"

        if cached_path.exists() and meta_path.exists():
            logger.info(f"Cache HIT for T_train={config.decoding.temperature} (key={key})")
            return cached_path

        logger.info(f"Cache MISS for T_train={config.decoding.temperature} (key={key})")
        return None

    def put(self, config: SynthesisConfig, output_path: Path) -> Path:
        """Store synthesis output in cache. Returns the cached path."""
        key = compute_cache_key(config)
        cached_path = self.cache_dir / f"synth_{key}.jsonl"
        meta_path = self.cache_dir / f"synth_{key}.meta.json"

        # Copy output to cache
        shutil.copy2(output_path, cached_path)

        # Save metadata
        meta = {
            "key": key,
            "model_id": config.model_id,
            "temperature": config.decoding.temperature,
            "top_k": config.decoding.top_k,
            "top_p": config.decoding.top_p,
            "config": config.model_dump(),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        logger.info(f"Cached synthesis output: {cached_path}")
        return cached_path

    def list_entries(self) -> list[dict]:
        """List all cached synthesis outputs."""
        entries = []
        for meta_path in sorted(self.cache_dir.glob("synth_*.meta.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                data_path = meta_path.with_suffix("").with_suffix(".jsonl")
                meta["_exists"] = data_path.exists()
                meta["_path"] = str(data_path)
                entries.append(meta)
            except Exception:
                continue
        return entries

    def clear(self) -> int:
        """Clear all cached entries. Returns number of entries removed."""
        count = 0
        for f in self.cache_dir.glob("synth_*"):
            f.unlink()
            count += 1
        return count
