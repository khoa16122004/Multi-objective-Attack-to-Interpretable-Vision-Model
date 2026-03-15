from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from algorithm import GA, NSGAII
from explain_method import get_gradcam_map, integrated_gradients, simple_gradient_map
from util import get_torchvision_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NSGA-II attack on multiple sampled images.")
    parser.add_argument(
        "--run-json-path",
        type=Path,
        required=True,
        help="Path to JSON in format: {class_id: [img_path, ...]}",
    )
    parser.add_argument("--num-runs", type=int, default=100, help="Number of samples to run.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--model", type=str, default="resnet50", help="Torchvision model name.")
    parser.add_argument("--dataset", type=str, default="imagenet", help="Dataset name.")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights.")

    parser.add_argument("--n-pix", type=int, default=40)
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--cr", type=float, default=0.9)
    parser.add_argument("--mu", type=float, default=0.9)
    parser.add_argument("--topk", type=int, default=10000)
    parser.add_argument(
        "--explain-method",
        type=str,
        default="gradcam",
        choices=["gradcam", "simple_gradient", "integrated_gradients"],
    )
    parser.add_argument("--ig-steps", type=int, default=5)
    parser.add_argument("--max-query", type=int, default=100000)

    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument(
        "--noise-mode",
        type=str,
        default="generation",
        choices=["fixed", "generation"],
    )
    parser.add_argument(
        "--intersec-mode",
        type=str,
        default="reference",
        choices=["reference", "target_region", "auto_low_region"],
    )
    parser.add_argument(
        "--target-region",
        type=str,
        default=None,
        help="Optional bbox string: y1,y2,x1,x2 (for target_region mode).",
    )
    parser.add_argument("--auto-region-percentile", type=float, default=30.0)
    parser.add_argument(
        "--target-objective",
        type=str,
        default="maximize_target_intersection",
        choices=["maximize_target_intersection", "maximize_target_importance"],
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/multi_sample"),
        help="Root output dir. Structure: output_root/<param_folder>/<class_id>/<image_name>/<sample_id>/...",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip sample if summary.json already exists.")

    parser.add_argument(
        "--algorithm",
        type=str,
        default="nsgaii",
        choices=["nsgaii", "ga"],
        help="Attack algorithm: nsgaii (multi-objective) or ga (single-objective, margin loss only).",
    )
    parser.add_argument(
        "--tournament-size",
        type=int,
        default=2,
        help="Tournament size for GA selection (only used when --algorithm=ga).",
    )
    return parser.parse_args()


def parse_target_region(region_str: str | None):
    if region_str is None:
        return None
    parts = [p.strip() for p in region_str.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--target-region must be in format 'y1,y2,x1,x2'")
    return tuple(int(v) for v in parts)


def _sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _build_param_folder(args: argparse.Namespace) -> str:
    parts = [
        f"algo={args.algorithm}",
        f"model={args.model}",
        f"method={args.explain_method}",
        f"n={args.n_pix}",
        f"pop={args.pop_size}",
        f"q={args.max_query}",
        f"topk={args.topk}",
        f"mode={args.intersec_mode}",
        f"obj={args.target_objective}",
        f"noise={args.noise_std}",
        f"seed={args.seed}",
    ]
    if args.intersec_mode == "target_region" and args.target_region is not None:
        parts.append(f"bbox={args.target_region.replace(',', '-')}")
    if args.intersec_mode == "auto_low_region":
        parts.append(f"p={args.auto_region_percentile}")
    return _sanitize("__".join(parts))


def _load_run_data(path: Path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("run-json must be dict: {class_id: [img_path, ...]}")

    out: Dict[str, List[str]] = {}
    for k, v in payload.items():
        if not isinstance(v, list):
            continue
        vals = [str(x) for x in v if isinstance(x, str) and len(str(x)) > 0]
        if vals:
            out[str(k)] = vals

    if len(out) == 0:
        raise ValueError("run-json contains no valid samples")
    return out


def _flatten_samples(run_data: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []
    for class_id, paths in sorted(run_data.items()):
        for p in paths:
            samples.append((class_id, p))
    return samples


def denorm_for_vis(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().clamp(0.0, 1.0)


def to_vis_space(x: torch.Tensor, normtransform):
    if normtransform is None:
        return denorm_for_vis(x)

    mean = torch.tensor(normtransform.mean, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(normtransform.std, device=x.device).view(1, -1, 1, 1)
    return denorm_for_vis(x * std + mean)


def get_explain_map_for_vis(model, model_name, x, label, method, ig_steps):
    method = method.lower()
    if method == "gradcam":
        m, _ = get_gradcam_map(model, model_name, x, target_class=label)
        return torch.as_tensor(m, device=x.device, dtype=torch.float32)
    if method == "simple_gradient":
        m, _ = simple_gradient_map(model, x, target_class=torch.tensor([label], device=x.device))
        return m
    if method == "integrated_gradients":
        m, _ = integrated_gradients(
            model,
            x,
            target_class=torch.tensor([label], device=x.device),
            steps=ig_steps,
        )
        return m
    raise ValueError("Unsupported explain method")


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    target_region = parse_target_region(args.target_region)

    run_data = _load_run_data(args.run_json_path)
    all_samples = _flatten_samples(run_data)

    if len(all_samples) == 0:
        raise RuntimeError("No sample found in run-json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sptransform, normtransform = get_torchvision_model(
        model_name=args.model,
        dataset_name=args.dataset,
        pretrained=args.pretrained,
    )
    model = model.to(device).eval()

    sample_count = min(args.num_runs, len(all_samples))
    chosen = random.sample(all_samples, k=sample_count) if sample_count < len(all_samples) else all_samples

    param_folder = _build_param_folder(args)
    run_root = args.output_root / param_folder
    run_root.mkdir(parents=True, exist_ok=True)

    run_report = {
        "run_json_path": str(args.run_json_path),
        "num_requested": int(args.num_runs),
        "num_selected": int(len(chosen)),
        "output_root": str(run_root),
        "runs": [],
    }

    for idx, (class_id, image_path_str) in enumerate(tqdm(chosen, desc="Running multi-sample attack"), start=1):
        sample_id = f"{idx:04d}"
        image_name = _sanitize(Path(image_path_str).stem)
        sample_dir = run_root / _sanitize(class_id) / image_name / sample_id
        summary_path = sample_dir / "summary.json"

        if args.skip_existing and summary_path.exists():
            run_report["runs"].append(
                {
                    "sample_id": sample_id,
                    "class_id": class_id,
                    "image_path": image_path_str,
                    "status": "skipped",
                }
            )
            continue

        sample_dir.mkdir(parents=True, exist_ok=True)

        image_path = Path(image_path_str)
        status = "success"
        err = ""

        try:
            img = Image.open(image_path).convert("RGB")
            oimg_vis = sptransform(img).unsqueeze(0).to(device)
            if normtransform is not None:
                oimg = normtransform(oimg_vis.squeeze(0)).unsqueeze(0)
            else:
                oimg = oimg_vis.clone()

            with torch.no_grad():
                logits = model(oimg)
                olabel = int(logits.argmax(dim=1).item())

            attacker_cls = GA if args.algorithm == "ga" else NSGAII
            attacker_kwargs = dict(
                model=model,
                model_name=args.model,
                n=args.n_pix,
                pop_size=args.pop_size,
                cr=args.cr,
                mu=args.mu,
                topk=args.topk,
                explain_method=args.explain_method,
                ig_steps=args.ig_steps,
                noise_std=args.noise_std,
                noise_mode=args.noise_mode,
                intersec_mode=args.intersec_mode,
                target_region=target_region,
                auto_region_percentile=args.auto_region_percentile,
                target_objective=args.target_objective,
                seed=args.seed + idx,
            )
            if args.algorithm == "ga":
                attacker_kwargs["tournament_size"] = args.tournament_size
            attacker = attacker_cls(**attacker_kwargs)

            adv, rank0_advs, population, objectives, nqry, rank0 = attacker.solve(
                oimg=oimg,
                olabel=olabel,
                max_query=args.max_query,
            )

            with torch.no_grad():
                adv_logits = model(adv)

            map_o = get_explain_map_for_vis(
                model,
                args.model,
                oimg,
                olabel,
                args.explain_method,
                args.ig_steps,
            )
            map_a = get_explain_map_for_vis(
                model,
                args.model,
                adv,
                olabel,
                args.explain_method,
                args.ig_steps,
            )

            adv_pred = int(adv_logits.argmax(dim=1).item())
            changed_pixels = int((adv[0] - oimg[0]).abs().sum(dim=0).gt(1e-8).sum().item())

            torch.save(
                {
                    "objectives": objectives,
                    "history": attacker.history,
                    "nqry": nqry,
                    "rank0": rank0,
                    "olabel": olabel,
                    "adv_pred": adv_pred,
                    "class_id": class_id,
                    "image_path": str(image_path),
                },
                sample_dir / "nsga2_sparse_stats.pt",
            )

            adv_vis = to_vis_space(adv, normtransform)
            plt.figure(figsize=(5, 5))
            plt.imshow(adv_vis[0].permute(1, 2, 0))
            plt.title(f"Adversarial Image (pred={adv_pred})")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(sample_dir / "nsga2_adv_image.png", dpi=200)
            plt.close()

            # Visualize saliency and image
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            axes[0, 0].imshow(denorm_for_vis(oimg_vis[0]).permute(1, 2, 0))
            axes[0, 0].set_title(f"Original (pred={olabel})")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(adv_vis[0].permute(1, 2, 0))
            axes[0, 1].set_title(f"Adversarial (pred={adv_pred})")
            axes[0, 1].axis("off")

            method_title = args.explain_method.replace("_", " ").title()
            axes[1, 0].imshow(map_o[0].detach().cpu().numpy(), cmap="hot")
            axes[1, 0].set_title(f"Original {method_title}")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(map_a[0].detach().cpu().numpy(), cmap="hot")
            axes[1, 1].set_title(f"Adversarial {method_title}")
            axes[1, 1].axis("off")

            plt.tight_layout()
            plt.savefig(sample_dir / "nsga2_sparse_result.png", dpi=200)
            plt.close()

            if len(attacker.history) > 0:
                q = [h["nqry"] for h in attacker.history]
                margin_curve = [h.get("best_margin", h["best_ce"]) for h in attacker.history]
                objective2_curve = [h.get("best_objective2", h["best_intersection"]) for h in attacker.history]

                objective2_title = "Best Objective-2 vs Query"
                objective2_ylabel = "Objective-2"
                if args.intersec_mode == "reference":
                    objective2_title = "Best Top-k Intersection vs Query"
                    objective2_ylabel = "Intersection Ratio"
                elif args.target_objective == "maximize_target_intersection":
                    objective2_title = "Best Target Intersection Objective vs Query"
                    objective2_ylabel = "Negative Intersection"
                elif args.target_objective == "maximize_target_importance":
                    objective2_title = "Best Target Importance Objective vs Query"
                    objective2_ylabel = "Negative Importance"

                fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
                axes2[0].plot(q, margin_curve, marker="", linewidth=1.5)
                axes2[0].set_title("Best Margin vs Query")
                axes2[0].set_xlabel("Queries")
                axes2[0].set_ylabel("Margin")
                axes2[0].grid(True, alpha=0.3)

                axes2[1].plot(q, objective2_curve, marker="", linewidth=1.5)
                axes2[1].set_title(objective2_title)
                axes2[1].set_xlabel("Queries")
                axes2[1].set_ylabel(objective2_ylabel)
                axes2[1].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(sample_dir / "nsga2_score_curves.png", dpi=200)
                plt.close()

            # Save all rank-0 adversarial images and saliency maps as a grid
            if len(rank0_advs) > 0:
                n_rank0 = len(rank0_advs)
                ncols = min(n_rank0, 5)
                nrows = (n_rank0 + ncols - 1) // ncols
                rank0_batch = torch.cat(rank0_advs, dim=0)
                with torch.no_grad():
                    rank0_preds = model(rank0_batch).argmax(dim=1).detach().cpu().numpy()

                fig_adv, axes_adv = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
                axes_adv = np.array(axes_adv).reshape(-1)
                for k, adv_k in enumerate(rank0_advs):
                    adv_k_vis = to_vis_space(adv_k, normtransform)
                    pred_k = int(rank0_preds[k])
                    margin_k = float(objectives[rank0[k], 0])
                    inter_k = float(objectives[rank0[k], 1])
                    axes_adv[k].imshow(adv_k_vis[0].permute(1, 2, 0).cpu())
                    axes_adv[k].set_title(f"pred={pred_k}\\nMargin={margin_k:.2f} Obj2={inter_k:.2f}", fontsize=7)
                    axes_adv[k].axis("off")
                for k in range(n_rank0, len(axes_adv)):
                    axes_adv[k].axis("off")
                fig_adv.suptitle(f"Rank-0 Adversarial Images ({n_rank0} individuals)", fontsize=10)
                plt.tight_layout()
                plt.savefig(sample_dir / "nsga2_rank0_advs.png", dpi=150)
                plt.close()

                fig_sal, axes_sal = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
                axes_sal = np.array(axes_sal).reshape(-1)
                method_title = args.explain_method.replace("_", " ").title()
                for k, adv_k in enumerate(rank0_advs):
                    sal_k = get_explain_map_for_vis(
                        model,
                        args.model,
                        adv_k,
                        olabel,
                        args.explain_method,
                        args.ig_steps,
                    )
                    inter_k = float(objectives[rank0[k], 1])
                    axes_sal[k].imshow(sal_k[0].detach().cpu().numpy(), cmap="hot")
                    axes_sal[k].set_title(f"{method_title}\\nInt={inter_k:.2f}", fontsize=7)
                    axes_sal[k].axis("off")
                for k in range(n_rank0, len(axes_sal)):
                    axes_sal[k].axis("off")
                fig_sal.suptitle(f"Rank-0 Saliency Maps ({n_rank0} individuals)", fontsize=10)
                plt.tight_layout()
                plt.savefig(sample_dir / "nsga2_rank0_saliency.png", dpi=150)
                plt.close()

            last_history = attacker.history[-1] if len(attacker.history) > 0 else None
            summary = {
                "sample_id": sample_id,
                "class_id": class_id,
                "image_name": image_name,
                "image_path": str(image_path),
                "model": args.model,
                "explain_method": args.explain_method,
                "intersec_mode": args.intersec_mode,
                "target_objective": args.target_objective,
                "target_region": target_region,
                "auto_region_percentile": args.auto_region_percentile,
                "queries": int(nqry),
                "original_label": int(olabel),
                "adversarial_label": int(adv_pred),
                "changed_pixels": int(changed_pixels),
                "rank0_size": int(len(rank0)),
                "population_size": int(population.shape[0]),
                "best_margin": float(last_history["best_margin"]) if last_history is not None else None,
                "best_ce": float(last_history["best_margin"]) if last_history is not None else None,
                "best_objective2": float(last_history["best_objective2"]) if last_history is not None else None,
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        except Exception as exc:
            status = "failed"
            err = f"{type(exc).__name__}: {exc}"

        run_report["runs"].append(
            {
                "sample_id": sample_id,
                "class_id": class_id,
                "image_name": image_name,
                "image_path": image_path_str,
                "status": status,
                "error": err,
            }
        )

    report_path = run_root / "run_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2, ensure_ascii=False)

    n_success = sum(1 for r in run_report["runs"] if r["status"] == "success")
    n_failed = sum(1 for r in run_report["runs"] if r["status"] == "failed")
    n_skipped = sum(1 for r in run_report["runs"] if r["status"] == "skipped")
    print(f"Saved report: {report_path}")
    print(f"Total={len(run_report['runs'])} success={n_success} failed={n_failed} skipped={n_skipped}")


if __name__ == "__main__":
    main()
