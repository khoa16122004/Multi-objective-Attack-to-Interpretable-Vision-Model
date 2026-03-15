from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle


# Fixed input folder (no argparse). Change this path when needed.
RESULT_DIR = Path(
	"/kaggle/input/datasets/khoatrnnht/mapr-test/Multi-objective-Attack-to-Interpretable-Vision-Model/"
	"samples_100class_results/model_resnet18__method_gradcam__n_500__pop_100__q_100000__topk_10000"
	"__mode_reference__obj_maximize_target_intersection__noise_1.0__seed_0"
)

# Output folder for analysis artifacts.
OUT_DIR = RESULT_DIR / "analysis_fixed"


def select_by_rule(objectives: np.ndarray) -> tuple[int, float, float, bool]:
	"""Pick one individual by rule:

	1) If there are successful attacks (margin < 0), choose one with smallest intersection.
	2) Otherwise choose one with smallest margin.
	"""
	if objectives.ndim != 2 or objectives.shape[1] < 2:
		raise ValueError("objectives must have shape [N, 2]")

	margin = objectives[:, 0]
	inter = objectives[:, 1]
	success_mask = margin < 0

	if np.any(success_mask):
		candidates = np.where(success_mask)[0]
		chosen_local = int(np.argmin(inter[candidates]))
		idx = int(candidates[chosen_local])
		return idx, float(margin[idx]), float(inter[idx]), True

	idx = int(np.argmin(margin))
	return idx, float(margin[idx]), float(inter[idx]), False


def stepwise_align(
	x_src: np.ndarray,
	y_src: np.ndarray,
	x_target: np.ndarray,
) -> np.ndarray:
	"""Piecewise-constant alignment from source queries to target query grid."""
	out = np.full((x_target.shape[0],), np.nan, dtype=np.float64)
	if x_src.size == 0:
		return out

	order = np.argsort(x_src)
	x = x_src[order]
	y = y_src[order]

	# Keep last value for duplicate query entries.
	uniq_x = []
	uniq_y = []
	for xi, yi in zip(x.tolist(), y.tolist()):
		if len(uniq_x) == 0 or xi != uniq_x[-1]:
			uniq_x.append(xi)
			uniq_y.append(yi)
		else:
			uniq_y[-1] = yi

	x = np.asarray(uniq_x, dtype=np.int64)
	y = np.asarray(uniq_y, dtype=np.float64)

	idx = np.searchsorted(x, x_target, side="right") - 1
	valid = idx >= 0
	out[valid] = y[idx[valid]]
	return out


def collect_stat_files(root: Path) -> list[Path]:
	files = sorted(root.rglob("*sparse_stats.pt"))
	return [p for p in files if p.is_file()]


def load_stats_payload(stat_path: Path):
	"""Load stats .pt payload across PyTorch versions.

	PyTorch >= 2.6 defaults to weights_only=True, which can fail for numpy-rich
	checkpoint dicts. For trusted local experiment outputs, we fall back to
	weights_only=False.
	"""
	try:
		return torch.load(stat_path, map_location="cpu")
	except pickle.UnpicklingError:
		return torch.load(stat_path, map_location="cpu", weights_only=False)


def analyze_result_folder(result_dir: Path, out_dir: Path) -> None:
	if not result_dir.exists():
		raise FileNotFoundError(f"Result folder not found: {result_dir}")

	stat_files = collect_stat_files(result_dir)
	if len(stat_files) == 0:
		raise RuntimeError(f"No '*sparse_stats.pt' files found under: {result_dir}")

	out_dir.mkdir(parents=True, exist_ok=True)

	per_run_queries = []
	per_run_margin_curve = []
	per_run_inter_curve = []
	per_run_success_curve = []
	final_margin = []
	final_inter = []
	final_success = []

	for stat_path in stat_files:
		payload = load_stats_payload(stat_path)
		objectives = np.asarray(payload.get("objectives", []), dtype=np.float64)
		history = payload.get("history", [])

		if objectives.size == 0:
			continue

		_, m_fin, i_fin, s_fin = select_by_rule(objectives)
		final_margin.append(m_fin)
		final_inter.append(i_fin)
		final_success.append(1.0 if s_fin else 0.0)

		q_list = []
		m_list = []
		i_list = []
		s_list = []

		for h in history:
			q = int(h.get("nqry", 0))
			rank0_obj = np.asarray(h.get("rank0_objectives", []), dtype=np.float64)
			if rank0_obj.size == 0:
				continue
			rank0_obj = rank0_obj.reshape(-1, 2)

			_, m_it, i_it, s_it = select_by_rule(rank0_obj)
			q_list.append(q)
			m_list.append(m_it)
			i_list.append(i_it)
			s_list.append(1.0 if s_it else 0.0)

		if len(q_list) == 0:
			# Fallback when history is empty: use final point.
			q_list = [int(payload.get("nqry", 0))]
			m_list = [m_fin]
			i_list = [i_fin]
			s_list = [1.0 if s_fin else 0.0]

		per_run_queries.append(np.asarray(q_list, dtype=np.int64))
		per_run_margin_curve.append(np.asarray(m_list, dtype=np.float64))
		per_run_inter_curve.append(np.asarray(i_list, dtype=np.float64))
		per_run_success_curve.append(np.asarray(s_list, dtype=np.float64))

	if len(per_run_queries) == 0:
		raise RuntimeError("No valid runs found in sparse_stats payloads.")

	q_grid = np.unique(np.concatenate(per_run_queries, axis=0))
	q_grid.sort()

	margin_mat = []
	inter_mat = []
	success_mat = []
	for q, m, i, s in zip(
		per_run_queries,
		per_run_margin_curve,
		per_run_inter_curve,
		per_run_success_curve,
	):
		margin_mat.append(stepwise_align(q, m, q_grid))
		inter_mat.append(stepwise_align(q, i, q_grid))
		success_mat.append(stepwise_align(q, s, q_grid))

	margin_mat = np.stack(margin_mat, axis=0)
	inter_mat = np.stack(inter_mat, axis=0)
	success_mat = np.stack(success_mat, axis=0)

	margin_mean = np.nanmean(margin_mat, axis=0)
	margin_std = np.nanstd(margin_mat, axis=0)
	inter_mean = np.nanmean(inter_mat, axis=0)
	inter_std = np.nanstd(inter_mat, axis=0)
	asr_curve = np.nanmean(success_mat, axis=0) * 100.0

	final_margin = np.asarray(final_margin, dtype=np.float64)
	final_inter = np.asarray(final_inter, dtype=np.float64)
	final_success = np.asarray(final_success, dtype=np.float64)

	asr_final = float(final_success.mean() * 100.0)
	summary_lines = [
		f"result_dir: {result_dir}",
		f"num_runs: {len(final_success)}",
		f"ASR_percent: {asr_final:.4f}",
		f"final_margin_mean: {final_margin.mean():.6f}",
		f"final_margin_std: {final_margin.std():.6f}",
		f"final_intersection_mean: {final_inter.mean():.6f}",
		f"final_intersection_std: {final_inter.std():.6f}",
	]
	(out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

	csv_data = np.column_stack(
		[
			q_grid,
			margin_mean,
			margin_std,
			inter_mean,
			inter_std,
			asr_curve,
		]
	)
	np.savetxt(
		out_dir / "curves_mean_std.csv",
		csv_data,
		delimiter=",",
		header="query,margin_mean,margin_std,intersection_mean,intersection_std,asr_percent",
		comments="",
	)

	fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

	axes[0].plot(q_grid, margin_mean, color="tab:blue", linewidth=2, label="Mean margin")
	axes[0].fill_between(
		q_grid,
		margin_mean - margin_std,
		margin_mean + margin_std,
		color="tab:blue",
		alpha=0.2,
		label="Std",
	)
	axes[0].axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.7)
	axes[0].set_ylabel("Margin loss")
	axes[0].set_title("Best-per-iteration margin curve (mean +/- std)")
	axes[0].grid(True, alpha=0.3)
	axes[0].legend(loc="best")

	axes[1].plot(q_grid, inter_mean, color="tab:red", linewidth=2, label="Mean intersection")
	axes[1].fill_between(
		q_grid,
		inter_mean - inter_std,
		inter_mean + inter_std,
		color="tab:red",
		alpha=0.2,
		label="Std",
	)
	axes[1].set_ylabel("Intersection")
	axes[1].set_title("Best-per-iteration intersection curve (mean +/- std)")
	axes[1].grid(True, alpha=0.3)
	axes[1].legend(loc="best")

	axes[2].plot(q_grid, asr_curve, color="tab:green", linewidth=2)
	axes[2].set_ylabel("ASR (%)")
	axes[2].set_xlabel("Query")
	axes[2].set_title("ASR over query (rule-based selected individual)")
	axes[2].set_ylim(0.0, 100.0)
	axes[2].grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig(out_dir / "curves_mean_std.png", dpi=180)
	plt.close(fig)

	print("=== Result Analysis Finished ===")
	print(f"Input folder: {result_dir}")
	print(f"Runs analyzed: {len(final_success)}")
	print(f"ASR: {asr_final:.2f}%")
	print(f"Final margin mean/std: {final_margin.mean():.6f} / {final_margin.std():.6f}")
	print(f"Final intersection mean/std: {final_inter.mean():.6f} / {final_inter.std():.6f}")
	print(f"Saved: {out_dir / 'summary.txt'}")
	print(f"Saved: {out_dir / 'curves_mean_std.csv'}")
	print(f"Saved: {out_dir / 'curves_mean_std.png'}")


if __name__ == "__main__":
	analyze_result_folder(RESULT_DIR, OUT_DIR)