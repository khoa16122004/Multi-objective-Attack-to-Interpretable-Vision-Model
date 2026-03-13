from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from algorithm import NSGAII
from explain_method import get_gradcam_map
from util import get_torchvision_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run sparse NSGA-II multi-objective attack test.")
    parser.add_argument("--image", type=str, required=True, help="Path to source image.")
    parser.add_argument("--model", type=str, default="resnet50", help="Torchvision model name.")
    parser.add_argument("--dataset", type=str, default="imagenet", help="Dataset name.")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-pix", type=int, default=20, help="Number of pixel positions to flip per individual init.")
    parser.add_argument("--pop-size", type=int, default=30)
    parser.add_argument("--cr", type=float, default=0.9)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument(
        "--explain-method",
        type=str,
        default="gradcam",
        choices=["gradcam", "simple_gradient", "integrated_gradients"],
        help="Explanation method for objective-2.",
    )
    parser.add_argument("--ig-steps", type=int, default=100, help="Used when explain-method=integrated_gradients.")
    parser.add_argument("--max-query", type=int, default=10000, help="Maximum number of queries.")
    parser.add_argument("--target-image", type=str, default="", help="Optional reference image path for sparse mixing.")
    parser.add_argument("--noise-std", type=float, default=0.8, help="Used when target image is not provided.")
    parser.add_argument("--out-dir", type=str, default="eval_test")
    return parser.parse_args()


def denorm_for_vis(x):
    return x.detach().cpu().clamp(0.0, 1.0)


def to_vis_space(x, normtransform):
    if normtransform is None:
        return denorm_for_vis(x)

    mean = torch.tensor(normtransform.mean, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(normtransform.std, device=x.device).view(1, -1, 1, 1)
    return denorm_for_vis(x * std + mean)


def main():
    args = parse_args()

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

    if args.target_image:
        timg_pil = Image.open(args.target_image).convert("RGB")
        timg_vis = sptransform(timg_pil).unsqueeze(0).to(device)
        if normtransform is not None:
            timg = normtransform(timg_vis.squeeze(0)).unsqueeze(0)
        else:
            timg = timg_vis.clone()
    else:
        noise = torch.randn_like(oimg) * float(args.noise_std)
        timg = (oimg + noise).clamp(oimg.min().item(), oimg.max().item())

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
        seed=args.seed,
    )

    adv, population, objectives, nqry, rank0 = attacker.solve(
        oimg=oimg,
        timg=timg,
        olabel=olabel,
        max_query=args.max_query,
    )

    cam_o, _ = get_gradcam_map(model, args.model, oimg, target_class=olabel)
    cam_a, adv_logits = get_gradcam_map(model, args.model, adv, target_class=olabel)

    adv_pred = int(adv_logits.argmax(dim=1).item())

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

    axes[1, 0].imshow(cam_o[0], cmap="hot")
    axes[1, 0].set_title("Original Grad-CAM")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cam_a[0], cmap="hot")
    axes[1, 1].set_title("Adversarial Grad-CAM")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "nsga2_sparse_result.png", dpi=200)
    plt.close()

    if len(attacker.history) > 0:
        q = [h["nqry"] for h in attacker.history]
        ce_curve = [h["best_ce"] for h in attacker.history]
        inter_curve = [h["best_intersection"] for h in attacker.history]

        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
        axes2[0].plot(q, ce_curve, marker="o", linewidth=1.5)
        axes2[0].set_title("Best CE vs Query")
        axes2[0].set_xlabel("Queries")
        axes2[0].set_ylabel("Cross-Entropy")
        axes2[0].grid(True, alpha=0.3)

        axes2[1].plot(q, inter_curve, marker="o", linewidth=1.5)
        axes2[1].set_title("Best Top-k Intersection vs Query")
        axes2[1].set_xlabel("Queries")
        axes2[1].set_ylabel("Intersection Ratio")
        axes2[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / "nsga2_score_curves.png", dpi=200)
        plt.close()

    print("=== NSGA-II Sparse Attack Finished ===")
    print("Queries:", nqry)
    print("Original label:", olabel)
    print("Adversarial prediction:", adv_pred)
    print("Population size:", len(population))
    print("Rank-0 size:", len(rank0))
    print("Saved:", out_dir / "nsga2_sparse_result.png")
    print("Saved:", out_dir / "nsga2_adv_image.png")
    if len(attacker.history) > 0:
        print("Saved:", out_dir / "nsga2_score_curves.png")
    print("Saved:", out_dir / "nsga2_sparse_stats.pt")


if __name__ == "__main__":
    main()
