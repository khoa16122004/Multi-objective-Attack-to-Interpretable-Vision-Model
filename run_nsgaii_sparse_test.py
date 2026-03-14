from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from algorithm import NSGAII
from explain_method import get_gradcam_map, simple_gradient_map, integrated_gradients
from util import get_torchvision_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run sparse NSGA-II multi-objective attack test.")
    parser.add_argument("--image", type=str, required=True, help="Path to source image.")
    parser.add_argument("--model", type=str, default="resnet50", help="Torchvision model name.")
    parser.add_argument("--dataset", type=str, default="imagenet", help="Dataset name.")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-pix", type=int, default=40, help="Number of pixel positions to flip per individual init.")
    parser.add_argument("--pop-size", type=int, default=50)
    parser.add_argument("--cr", type=float, default=0.9)
    parser.add_argument("--mu", type=float, default=0.9)
    parser.add_argument("--topk", type=int, default=10000)
    parser.add_argument(
        "--explain-method",
        type=str,
        default="gradcam",
        choices=["gradcam", "simple_gradient", "integrated_gradients"],
        help="Explanation method for objective-2.",
    )
    parser.add_argument("--ig-steps", type=int, default=5, help="Used when explain-method=integrated_gradients.")
    parser.add_argument("--max-query", type=int, default=100000, help="Maximum number of queries.")
    parser.add_argument("--noise-std", type=float, default=1.0, help="Std of Gaussian noise on selected sparse pixels.")
    parser.add_argument(
        "--noise-mode",
        type=str,
        default="generation",
        choices=["fixed", "generation"],
        help="Noise sampling mode: fixed once per run, or resample each generation.",
    )
    parser.add_argument(
        "--intersec-mode",
        type=str,
        default="reference",
        choices=["reference", "target_region", "auto_low_region"],
        help="Objective-2 mode: original reference overlap, manual target region, or auto low-saliency region.",
    )
    parser.add_argument(
        "--target-region",
        type=str,
        default=None,
        help="Target region bbox as 'y1,y2,x1,x2' when intersec-mode=target_region.",
    )
    parser.add_argument(
        "--auto-region-percentile",
        type=float,
        default=30.0,
        help="Percentile for auto_low_region (lower saliency area selection).",
    )
    parser.add_argument(
        "--target-objective",
        type=str,
        default="maximize_target_intersection",
        choices=["maximize_target_intersection", "maximize_target_importance"],
        help="For target-region modes: maximize top-k overlap with target map, or maximize saliency importance in target region.",
    )
    parser.add_argument("--out-dir", type=str, default="eval_test")
    return parser.parse_args()


def parse_target_region(region_str):
    if region_str is None:
        return None
    parts = [p.strip() for p in region_str.split(",") if p.strip() != ""]
    if len(parts) != 4:
        raise ValueError("--target-region must be in format 'y1,y2,x1,x2'")
    return tuple(int(v) for v in parts)


def denorm_for_vis(x):
    return x.detach().cpu().clamp(0.0, 1.0)


def to_vis_space(x, normtransform):
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


def main():
    args = parse_args()
    target_region = parse_target_region(args.target_region)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, sptransform, normtransform = get_torchvision_model(
        model_name=args.model,
        dataset_name=args.dataset,
        pretrained=args.pretrained,
    )
    model = model.to(device).eval()

    img = Image.open(args.image).convert("RGB")
    oimg_vis = sptransform(img).unsqueeze(0).to(device)
    if normtransform is not None:
        oimg = normtransform(oimg_vis.squeeze(0)).unsqueeze(0)
    else:
        oimg = oimg_vis.clone()

    with torch.no_grad():
        logits = model(oimg)
        olabel = int(logits.argmax(dim=1).item())

    attacker = NSGAII(
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
        seed=args.seed,
    )

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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save tensors
    torch.save(
        {
            "objectives": objectives,
            "history": attacker.history,
            "nqry": nqry,
            "rank0": rank0,
            "olabel": olabel,
            "adv_pred": adv_pred,
        },
        out_dir / "nsga2_sparse_stats.pt",
    )

    adv_vis = to_vis_space(adv, normtransform)
    plt.figure(figsize=(5, 5))
    plt.imshow(adv_vis[0].permute(1, 2, 0))
    plt.title(f"Adversarial Image (pred={adv_pred})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "nsga2_adv_image.png", dpi=200)
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
    plt.savefig(out_dir / "nsga2_sparse_result.png", dpi=200)
    plt.close()

    if len(attacker.history) > 0:
        q = [h["nqry"] for h in attacker.history]
        ce_curve = [h["best_ce"] for h in attacker.history]
        inter_curve = [h["best_intersection"] for h in attacker.history]

        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
        axes2[0].plot(q, ce_curve, marker="", linewidth=1.5)
        axes2[0].set_title("Best CE vs Query")
        axes2[0].set_xlabel("Queries")
        axes2[0].set_ylabel("Cross-Entropy")
        axes2[0].grid(True, alpha=0.3)

        axes2[1].plot(q, inter_curve, marker="", linewidth=1.5)
        axes2[1].set_title("Best Top-k Intersection vs Query")
        axes2[1].set_xlabel("Queries")
        axes2[1].set_ylabel("Intersection Ratio")
        axes2[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "nsga2_score_curves.png", dpi=200)
        plt.close()

    # Save all rank-0 adversarial images and saliency maps as a grid
    if len(rank0_advs) > 0:
        n_rank0 = len(rank0_advs)
        ncols = min(n_rank0, 5)
        nrows = (n_rank0 + ncols - 1) // ncols
        rank0_batch = torch.cat(rank0_advs, dim=0)
        with torch.no_grad():
            rank0_preds = model(rank0_batch).argmax(dim=1).detach().cpu().numpy()

        # Grid: adv images
        fig_adv, axes_adv = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axes_adv = np.array(axes_adv).reshape(-1)
        for k, adv_k in enumerate(rank0_advs):
            adv_k_vis = to_vis_space(adv_k, normtransform)
            pred_k = int(rank0_preds[k])
            ce_k = float(objectives[rank0[k], 0])
            inter_k = float(objectives[rank0[k], 1])
            axes_adv[k].imshow(adv_k_vis[0].permute(1, 2, 0).cpu())
            axes_adv[k].set_title(f"pred={pred_k}\nCE={ce_k:.2f} Int={inter_k:.2f}", fontsize=7)
            axes_adv[k].axis("off")
        for k in range(n_rank0, len(axes_adv)):
            axes_adv[k].axis("off")
        fig_adv.suptitle(f"Rank-0 Adversarial Images ({n_rank0} individuals)", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / "nsga2_rank0_advs.png", dpi=150)
        plt.close()

        # Grid: saliency maps
        fig_sal, axes_sal = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axes_sal = np.array(axes_sal).reshape(-1)
        method_title = args.explain_method.replace("_", " ").title()
        for k, adv_k in enumerate(rank0_advs):
            sal_k = get_explain_map_for_vis(
                model, args.model, adv_k, olabel, args.explain_method, args.ig_steps
            )
            inter_k = float(objectives[rank0[k], 1])
            axes_sal[k].imshow(sal_k[0].detach().cpu().numpy(), cmap="hot")
            axes_sal[k].set_title(f"{method_title}\nInt={inter_k:.2f}", fontsize=7)
            axes_sal[k].axis("off")
        for k in range(n_rank0, len(axes_sal)):
            axes_sal[k].axis("off")
        fig_sal.suptitle(f"Rank-0 Saliency Maps ({n_rank0} individuals)", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / "nsga2_rank0_saliency.png", dpi=150)
        plt.close()

    print("=== NSGA-II Sparse Attack Finished ===")
    print("Queries:", nqry)
    print("Original label:", olabel)
    print("Adversarial prediction:", adv_pred)
    print("Changed pixels:", changed_pixels)
    print("Population size:", len(population))
    print("Rank-0 size:", len(rank0))
    print("Saved:", out_dir / "nsga2_sparse_result.png")
    print("Saved:", out_dir / "nsga2_adv_image.png")
    if len(rank0_advs) > 0:
        print("Saved:", out_dir / "nsga2_rank0_advs.png")
        print("Saved:", out_dir / "nsga2_rank0_saliency.png")
    if len(attacker.history) > 0:
        print("Saved:", out_dir / "nsga2_score_curves.png")
    print("Saved:", out_dir / "nsga2_sparse_stats.pt")


if __name__ == "__main__":
    main()
