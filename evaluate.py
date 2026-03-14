from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image

from util import get_torchvision_model


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _forward_logits(model: torch.nn.Module, input_batch: torch.Tensor) -> torch.Tensor:
	out = model(input_batch)
	if isinstance(out, torch.Tensor):
		return out
	if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
		return out[0]
	raise TypeError(f"Unsupported model output type: {type(out)}")


def _load_class_key_to_idx(mapping_file: Path) -> Dict[str, int]:
	# Label file format in this repo: {"n01440764": [0, "tench"], ...}
	with open(mapping_file, "r", encoding="utf-8") as f:
		raw = json.load(f)

	if not isinstance(raw, dict):
		raise ValueError("class_index_json must be a dict mapping class folder key -> class index")

	mapping: Dict[str, int] = {}
	for k, v in raw.items():
		if isinstance(v, int):
			mapping[str(k)] = int(v)
		elif isinstance(v, (list, tuple)) and len(v) > 0:
			mapping[str(k)] = int(v[0])
		elif isinstance(v, dict) and "index" in v:
			mapping[str(k)] = int(v["index"])

	if len(mapping) == 0:
		raise ValueError("Could not build mapping from class_index_json")
	return mapping


def _collect_samples(
	imagenet_val_dir: Path,
	key_to_idx: Dict[str, int],
) -> List[Tuple[str, int, Path]]:
	samples: List[Tuple[str, int, Path]] = []
	class_dirs = sorted(p for p in imagenet_val_dir.iterdir() if p.is_dir())

	for class_dir in class_dirs:
		class_key = class_dir.name
		if class_key not in key_to_idx:
			continue
		gt_idx = int(key_to_idx[class_key])

		for img_path in sorted(class_dir.iterdir()):
			if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTS:
				continue
			samples.append((class_key, gt_idx, img_path))

	return samples


def _batched(items: List[Tuple[str, int, Path]], batch_size: int) -> Iterable[List[Tuple[str, int, Path]]]:
	for i in range(0, len(items), batch_size):
		yield items[i : i + batch_size]


def evaluate_correct_samples(
	model: torch.nn.Module,
	spatial_transform,
	normalize_transform,
	imagenet_val_dir: Path,
	class_index_json: Path,
	batch_size: int = 64,
) -> Dict[str, List[str]]:
	key_to_idx = _load_class_key_to_idx(class_index_json)
	samples = _collect_samples(imagenet_val_dir, key_to_idx)
	if len(samples) == 0:
		raise RuntimeError("No valid samples found. Check val dir and class-index mapping.")

	device = next(model.parameters()).device
	model.eval()

	# Keep class_id as folder key (wnid), as requested.
	correct_by_class: Dict[str, List[str]] = {}

	with torch.no_grad():
		for batch in _batched(samples, batch_size):
			x_batch = []
			metas = []
			for class_key, gt_idx, img_path in batch:
				img = Image.open(img_path).convert("RGB")
				x = spatial_transform(img)
				if normalize_transform is not None:
					x = normalize_transform(x)
				x_batch.append(x)
				metas.append((class_key, gt_idx, img_path))

			inp = torch.stack(x_batch, dim=0).to(device)
			logits = _forward_logits(model, inp)
			preds = logits.argmax(dim=1).detach().cpu().tolist()

			for (class_key, gt_idx, img_path), pred in zip(metas, preds):
				if int(pred) == int(gt_idx):
					correct_by_class.setdefault(class_key, []).append(str(img_path))

	return correct_by_class


def _sanitize_filename(value: str) -> str:
	return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def save_correct_samples_json(correct_by_class: Dict[str, List[str]], output_json: Path) -> None:
	output_json.parent.mkdir(parents=True, exist_ok=True)
	with open(output_json, "w", encoding="utf-8") as f:
		json.dump(correct_by_class, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate model(s) and save correctly predicted samples by class folder key."
	)
	parser.add_argument(
		"--imagenet-val-dir",
		type=Path,
		default=Path("/datastore/elo/quanphm/dataset/ImageNet1K/val"),
	)
	parser.add_argument(
		"--class-index-json",
		type=Path,
		default=Path("data/imagenet_1k_label.json"),
	)
	parser.add_argument(
		"--models",
		nargs="+",
		default=["resnet50"],
		help="One or more torchvision model names.",
	)
	parser.add_argument("--dataset", type=str, default="imagenet")
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("correct_runs"),
		help="Output directory. Each model writes one json: <model_name>.json",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if not args.imagenet_val_dir.exists():
		raise FileNotFoundError(f"imagenet_val_dir not found: {args.imagenet_val_dir}")
	if not args.class_index_json.exists():
		raise FileNotFoundError(f"class_index_json not found: {args.class_index_json}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.output_dir.mkdir(parents=True, exist_ok=True)

	for model_name in args.models:
		model, spatial_transform, normalize_transform = get_torchvision_model(
			model_name=model_name,
			dataset_name=args.dataset,
			pretrained=True,
		)
		model = model.to(device)

		result = evaluate_correct_samples(
			model=model,
			spatial_transform=spatial_transform,
			normalize_transform=normalize_transform,
			imagenet_val_dir=args.imagenet_val_dir,
			class_index_json=args.class_index_json,
			batch_size=args.batch_size,
		)

		out_json = args.output_dir / f"{_sanitize_filename(model_name)}.json"
		save_correct_samples_json(result, out_json)

		total_correct = sum(len(v) for v in result.values())
		print(f"[{model_name}] saved: {out_json}")
		print(f"[{model_name}] classes with correct samples: {len(result)}")
		print(f"[{model_name}] total correct samples: {total_correct}")


if __name__ == "__main__":
	main()
