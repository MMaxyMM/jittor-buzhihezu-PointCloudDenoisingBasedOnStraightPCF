#!/usr/bin/env python3
"""Create a deterministic stratified local train/test split using symlinks."""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path


OBJ_RELATIVE_PATH = Path("models/model_normalized.obj")


def category_seed(seed: int, synset_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{synset_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def discover_models(source_root: Path, synset_id: str):
    category_root = source_root / synset_id
    models = []
    for model_root in sorted(path for path in category_root.iterdir() if path.is_dir()):
        if (model_root / OBJ_RELATIVE_PATH).is_file():
            models.append(model_root)
    return models


def make_model_link(destination_root: Path, model_root: Path, dataset_root: Path):
    synset_id = model_root.parent.name
    link = destination_root / "shapenet" / synset_id / model_root.name
    link.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(model_root, start=link.parent)
    link.symlink_to(target, target_is_directory=True)


def write_lines(path: Path, lines):
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="dataset_train")
    parser.add_argument("--test_ratio", type=float, default=0.02)
    parser.add_argument("--min_test_per_category", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "删除并重建已有local_train/local_test软链接目录；"
            "不会删除dataset_train/shapenet中的原始模型"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_dir).resolve()
    source_root = dataset_root / "shapenet"
    train_root = dataset_root / "local_train"
    test_root = dataset_root / "local_test"

    if not source_root.is_dir():
        raise SystemExit(f"source directory does not exist: {source_root}")
    existing_roots = [root for root in (train_root, test_root) if root.exists()]
    if existing_roots and not args.overwrite:
        raise SystemExit(
            "local split already exists; rerun with --overwrite to replace only "
            "the local_train/local_test link trees: "
            + ", ".join(str(root) for root in existing_roots)
        )
    if args.overwrite:
        for root in existing_roots:
            if root.is_symlink():
                root.unlink()
            else:
                shutil.rmtree(root)
    if not 0.0 < args.test_ratio < 1.0:
        raise SystemExit("--test_ratio must be between 0 and 1")
    if args.min_test_per_category < 1:
        raise SystemExit("--min_test_per_category must be positive")

    train_entries = []
    test_entries = []
    summary = {}

    synset_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    for synset_dir in synset_dirs:
        synset_id = synset_dir.name
        models = discover_models(source_root, synset_id)
        if not models:
            continue

        test_count = max(
            args.min_test_per_category,
            math.ceil(len(models) * args.test_ratio),
        )
        if test_count >= len(models):
            raise SystemExit(
                f"category {synset_id} has {len(models)} models, but "
                f"{test_count} were requested for testing"
            )

        rng = random.Random(category_seed(args.seed, synset_id))
        shuffled = models[:]
        rng.shuffle(shuffled)
        test_names = {path.name for path in shuffled[:test_count]}

        category_train = 0
        category_test = 0
        for model_root in models:
            relative_entry = (
                Path("shapenet") / synset_id / model_root.name
            ).as_posix()
            if model_root.name in test_names:
                make_model_link(test_root, model_root, dataset_root)
                test_entries.append(relative_entry)
                category_test += 1
            else:
                make_model_link(train_root, model_root, dataset_root)
                train_entries.append(relative_entry)
                category_train += 1

        summary[synset_id] = {
            "total": len(models),
            "local_train": category_train,
            "local_test": category_test,
        }

    train_entries.sort()
    test_entries.sort()
    write_lines(train_root / "datalist.txt", train_entries)
    write_lines(test_root / "datalist.txt", test_entries)

    manifest = {
        "dataset_dir": str(dataset_root),
        "source": "shapenet",
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "min_test_per_category": args.min_test_per_category,
        "rounding": "ceil",
        "storage": "relative directory symlinks; OBJ files are not copied",
        "total": len(train_entries) + len(test_entries),
        "local_train": len(train_entries),
        "local_test": len(test_entries),
        "categories": summary,
    }
    (dataset_root / "local_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
