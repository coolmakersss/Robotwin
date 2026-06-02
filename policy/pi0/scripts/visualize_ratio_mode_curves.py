import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DEFAULT_DATA_ROOT = (
    "/mnt/afs/huangdi/xiangenda/RoboTwin/policy/pi0/"
    "training_data_50_tasks_cts_10d_action_mode_ratio"
)
DEFAULT_OUTPUT_DIR = "/mnt/afs/huangdi/xiangenda/mode_ratio_check/figures"
DEFAULT_SAMPLE_FRAMES_PER_TASK = 16

DEFAULT_SELECTED_EPISODES = {
    "grab_roller": {
        "task_dir": "grab_roller-aloha-agilex_clean_50-50",
        "episode": 49,
        "expected_switch_frames": [40],
    },
    "lift_pot": {
        "task_dir": "lift_pot-aloha-agilex_clean_50-50",
        "episode": 33,
        "expected_switch_frames": [61],
    },
    "handover_mic": {
        "task_dir": "handover_mic-aloha-agilex_clean_50-50",
        "episode": 0,
        "expected_switch_frames": [107, 164],
    },
    "shake_bottle": {
        "task_dir": "shake_bottle-aloha-agilex_clean_50-50",
        "episode": 30,
        "expected_switch_frames": [36, 222],
    },
}

DEFAULT_MODE_NAMES = (
    "stabilize",
    "absolute",
    "relative_translation_close",
    "relative_translation_apart",
    "relative_rotation",
    "mixed",
)
DEFAULT_RATIO_LABELS = ("absolute", "relative_translation", "relative_rotation")

RATIO_COLORS = {
    "absolute": "#2C6BED",
    "relative_translation": "#16A34A",
    "relative_rotation": "#C026D3",
}
MODE_COLORS = {
    "stabilize": "#A3A3A3",
    "absolute": "#93C5FD",
    "relative_translation": "#86EFAC",
    "relative_translation_close": "#86EFAC",
    "relative_translation_apart": "#67E8F9",
    "relative_rotation": "#E9D5FF",
    "mixed": "#FDE68A",
}


def read_json_attr(attrs, name, default):
    if name not in attrs:
        return tuple(default)
    value = attrs[name]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return tuple(json.loads(value))


def mode_name(mode_id, mode_names):
    mode_id = int(mode_id)
    if 0 <= mode_id < len(mode_names):
        return mode_names[mode_id]
    return f"unknown_{mode_id}"


def mode_color(mode_id, mode_names):
    name = mode_name(mode_id, mode_names)
    return MODE_COLORS.get(name, "#DDDDDD")


def display_mode_name(name):
    replacements = {
        "relative_translation_close": "rel_trans_close",
        "relative_translation_apart": "rel_trans_apart",
        "relative_rotation": "rel_rot",
    }
    return replacements.get(name, name)


def display_ratio_label(label):
    replacements = {
        "absolute": "abs",
        "relative_translation": "rel_trans",
        "relative_rotation": "rel_rot",
    }
    return replacements.get(label, label)


def get_switch_frames(modes):
    if len(modes) <= 1:
        return []
    return (np.flatnonzero(modes[1:] != modes[:-1]) + 1).astype(int).tolist()


def get_mode_segments(modes, mode_names, ratios, ratio_mask):
    switch_frames = get_switch_frames(modes)
    bounds = [0] + switch_frames + [len(modes)]
    segments = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        mode_id = int(modes[start])
        valid = ratio_mask[start:end] > 0.0
        if np.any(valid):
            mean_ratio = ratios[start:end][valid].mean(axis=0).astype(float).tolist()
        else:
            mean_ratio = [0.0] * ratios.shape[1]
        segments.append({
            "start_frame": int(start),
            "end_frame": int(end - 1),
            "length": int(end - start),
            "mode": mode_id,
            "mode_name": mode_name(mode_id, mode_names),
            "mean_ratio": mean_ratio,
        })
    return segments


def load_episode(hdf5_path):
    with h5py.File(hdf5_path, "r") as h5:
        ratios = np.asarray(h5["/ratio"][:], dtype=np.float32)
        modes = np.asarray(h5["/mode"][:], dtype=np.int32)
        ratio_mask = np.asarray(h5["/ratio_mask"][:], dtype=np.float32)
        mode_names = read_json_attr(h5.attrs, "mode_names", DEFAULT_MODE_NAMES)
        ratio_labels = read_json_attr(h5.attrs, "ratio_labels", DEFAULT_RATIO_LABELS)
    return ratios, modes, ratio_mask, mode_names, ratio_labels


def annotate_mode_spans(ax, modes, mode_names, ratios, ratio_mask):
    segments = get_mode_segments(modes, mode_names, ratios, ratio_mask)
    used_modes = []
    for segment in segments:
        start = segment["start_frame"]
        end = segment["end_frame"]
        mode_id = segment["mode"]
        name = segment["mode_name"]
        ax.axvspan(start - 0.5, end + 0.5, color=mode_color(mode_id, mode_names), alpha=0.20, lw=0)
        center = (start + end) / 2.0
        ax.text(
            center,
            1.045,
            display_mode_name(name),
            ha="center",
            va="bottom",
            fontsize=18.0,
            color="#222222",
            clip_on=False,
        )
        if mode_id not in used_modes:
            used_modes.append(mode_id)
    return segments, used_modes


def plot_ratio_mode_curve(task, episode, hdf5_path, output_dir, dpi):
    ratios, modes, ratio_mask, mode_names, ratio_labels = load_episode(hdf5_path)
    if ratios.shape[0] != modes.shape[0]:
        raise ValueError(f"ratio/mode length mismatch in {hdf5_path}: {ratios.shape[0]} vs {modes.shape[0]}")

    os.makedirs(output_dir, exist_ok=True)
    x = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(17.5, 5.5))

    segments, _ = annotate_mode_spans(ax, modes, mode_names, ratios, ratio_mask)

    for idx, label in enumerate(ratio_labels):
        color = RATIO_COLORS.get(label, None)
        ax.plot(x, ratios[:, idx], label=display_ratio_label(label), color=color, linewidth=3.0)

    switch_frames = get_switch_frames(modes)
    for frame_idx in switch_frames:
        ax.axvline(frame_idx, color="#111111", linestyle="--", linewidth=1.2, alpha=0.70)
        ax.text(
            frame_idx + 1,
            0.96,
            f"f={frame_idx}",
            rotation=90,
            ha="left",
            va="top",
            fontsize=18.0,
            color="#111111",
        )

    ax.set_xlabel("Frame index", fontsize=22)
    ax.set_ylabel("Ratio value", fontsize=22)
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlim(0, max(len(modes) - 1, 1))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(axis="both", labelsize=20)

    fig.tight_layout()
    base_name = f"{task}_episode_{episode}_ratio_mode_curve"
    png_path = os.path.join(output_dir, f"{base_name}.png")
    pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "curve_png": png_path,
        "curve_pdf": pdf_path,
        "switch_frames": switch_frames,
        "mode_segments": segments,
        "mode_names": list(mode_names),
        "ratio_labels": list(ratio_labels),
    }


def save_standalone_legend(output_dir, ratio_labels, mode_names, mode_ids, dpi):
    fig, ax = plt.subplots(figsize=(3.0, 10.8))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(0.06, 0.96, "Ratio", fontsize=22, fontweight="bold", va="top", color="#111111")
    ratio_y = 0.86
    for label in ratio_labels:
        ax.plot([0.08, 0.34], [ratio_y, ratio_y], color=RATIO_COLORS.get(label, "#333333"), linewidth=5.0)
        ax.text(0.42, ratio_y, display_ratio_label(label), fontsize=19, va="center", color="#111111")
        ratio_y -= 0.095

    ax.text(0.06, 0.58, "Mode", fontsize=22, fontweight="bold", va="top", color="#111111")
    mode_y = 0.48
    for mode_id in mode_ids:
        ax.add_patch(Rectangle((0.08, mode_y - 0.026), 0.24, 0.052, color=mode_color(mode_id, mode_names), alpha=0.72, lw=0))
        ax.text(
            0.42,
            mode_y,
            display_mode_name(mode_name(mode_id, mode_names)),
            fontsize=19,
            va="center",
            color="#111111",
        )
        mode_y -= 0.090

    png_path = os.path.join(output_dir, "ratio_mode_legend.png")
    pdf_path = os.path.join(output_dir, "ratio_mode_legend.pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", transparent=True)
    fig.savefig(pdf_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return {"png": png_path, "pdf": pdf_path}


def decode_cam_high_for_write(h5, frame_idx, *, swap_rb):
    encoded = h5["/observations/images/cam_high"][frame_idx]
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to decode cam_high frame {frame_idx}.")
    if swap_rb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def save_keyframes(task, episode, hdf5_path, output_dir, switch_frames, *, swap_rb):
    keyframe_dir = os.path.join(output_dir, "keyframes")
    os.makedirs(keyframe_dir, exist_ok=True)
    saved = []
    with h5py.File(hdf5_path, "r") as h5:
        modes = np.asarray(h5["/mode"][:], dtype=np.int32)
        mode_names = read_json_attr(h5.attrs, "mode_names", DEFAULT_MODE_NAMES)
        for frame_idx in switch_frames:
            if frame_idx <= 0 or frame_idx >= len(modes):
                continue
            prev_mode = mode_name(modes[frame_idx - 1], mode_names)
            next_mode = mode_name(modes[frame_idx], mode_names)
            image = decode_cam_high_for_write(h5, frame_idx, swap_rb=swap_rb)
            filename = (
                f"{task}_episode_{episode}_frame_{frame_idx:04d}_"
                f"{safe_name(prev_mode)}_to_{safe_name(next_mode)}.png"
            )
            path = os.path.join(keyframe_dir, filename)
            if not cv2.imwrite(path, image):
                raise RuntimeError(f"Failed to save keyframe: {path}")
            saved.append({
                "frame": int(frame_idx),
                "from_mode": prev_mode,
                "to_mode": next_mode,
                "path": path,
            })
    return saved


def select_uniform_frames(num_frames, sample_count, excluded_frames):
    if sample_count <= 0 or num_frames <= 0:
        return []
    excluded = set(int(frame) for frame in excluded_frames)
    available = [frame for frame in range(num_frames) if frame not in excluded]
    if not available:
        return []
    count = min(sample_count, len(available))
    positions = np.linspace(0, len(available) - 1, count)
    selected = []
    for position in positions:
        frame = available[int(round(float(position)))]
        if frame not in selected:
            selected.append(frame)
    if len(selected) < count:
        for frame in available:
            if frame not in selected:
                selected.append(frame)
                if len(selected) == count:
                    break
    return sorted(selected)


def save_sample_frames(task, episode, hdf5_path, output_dir, switch_frames, sample_count, *, swap_rb):
    sample_dir = os.path.join(output_dir, "sample_frames")
    os.makedirs(sample_dir, exist_ok=True)
    saved = []
    with h5py.File(hdf5_path, "r") as h5:
        modes = np.asarray(h5["/mode"][:], dtype=np.int32)
        mode_names = read_json_attr(h5.attrs, "mode_names", DEFAULT_MODE_NAMES)
        selected_frames = select_uniform_frames(len(modes), sample_count, switch_frames)
        for frame_idx in selected_frames:
            current_mode = mode_name(modes[frame_idx], mode_names)
            image = decode_cam_high_for_write(h5, frame_idx, swap_rb=swap_rb)
            filename = f"{task}_episode_{episode}_frame_{frame_idx:04d}_{safe_name(current_mode)}.png"
            path = os.path.join(sample_dir, filename)
            if not cv2.imwrite(path, image):
                raise RuntimeError(f"Failed to save sample frame: {path}")
            saved.append({
                "frame": int(frame_idx),
                "mode": current_mode,
                "path": path,
            })
    return saved


def hdf5_path_for(data_root, task_dir, episode):
    return os.path.join(data_root, task_dir, f"episode_{episode}", f"episode_{episode}.hdf5")


def visualize_selected(data_root, output_dir, dpi, strict_switch_frames, sample_frames_per_task, swap_rb):
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "data_root": data_root,
        "output_dir": output_dir,
        "sample_frames_per_task": int(sample_frames_per_task),
        "swap_rb_before_cv2_write": bool(swap_rb),
        "tasks": {},
    }
    legend_mode_ids = []
    legend_mode_names = DEFAULT_MODE_NAMES
    legend_ratio_labels = DEFAULT_RATIO_LABELS

    for task, spec in DEFAULT_SELECTED_EPISODES.items():
        episode = int(spec["episode"])
        hdf5_path = hdf5_path_for(data_root, spec["task_dir"], episode)
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(hdf5_path)

        curve_info = plot_ratio_mode_curve(task, episode, hdf5_path, output_dir, dpi)
        expected = list(spec["expected_switch_frames"])
        actual = curve_info["switch_frames"]
        if strict_switch_frames and actual != expected:
            raise ValueError(f"{task} episode_{episode} switch frames mismatch: expected {expected}, got {actual}")

        keyframes = save_keyframes(task, episode, hdf5_path, output_dir, actual, swap_rb=swap_rb)
        sample_frames = save_sample_frames(
            task,
            episode,
            hdf5_path,
            output_dir,
            actual,
            sample_frames_per_task,
            swap_rb=swap_rb,
        )
        legend_mode_names = tuple(curve_info["mode_names"])
        legend_ratio_labels = tuple(curve_info["ratio_labels"])
        for segment in curve_info["mode_segments"]:
            mode_id = int(segment["mode"])
            if mode_id not in legend_mode_ids:
                legend_mode_ids.append(mode_id)
        summary["tasks"][task] = {
            "episode": episode,
            "hdf5_path": hdf5_path,
            "expected_switch_frames": expected,
            "switch_frames": actual,
            "curve_png": curve_info["curve_png"],
            "curve_pdf": curve_info["curve_pdf"],
            "keyframes": keyframes,
            "sample_frames": sample_frames,
            "mode_segments": curve_info["mode_segments"],
            "mode_names": curve_info["mode_names"],
            "ratio_labels": curve_info["ratio_labels"],
        }

    legend_mode_ids_for_plot = list(legend_mode_ids)
    if legend_mode_names and legend_mode_names[0] == "stabilize" and 0 not in legend_mode_ids_for_plot:
        legend_mode_ids_for_plot.append(0)

    summary["legend"] = save_standalone_legend(
        output_dir,
        legend_ratio_labels,
        legend_mode_names,
        legend_mode_ids_for_plot,
        dpi,
    )

    summary_path = os.path.join(output_dir, "ratio_mode_curve_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_path, summary


def main():
    parser = argparse.ArgumentParser(description="Plot ratio curves with mode segments and save mode-switch keyframes.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--sample-frames-per-task", type=int, default=DEFAULT_SAMPLE_FRAMES_PER_TASK)
    parser.add_argument(
        "--no-rb-swap",
        action="store_true",
        help="Disable the default R/B channel swap before writing extracted frames.",
    )
    parser.add_argument(
        "--allow-switch-mismatch",
        action="store_true",
        help="Do not raise if the selected episode switch frames differ from the expected low-switch plan.",
    )
    args = parser.parse_args()

    summary_path, summary = visualize_selected(
        args.data_root,
        args.output_dir,
        args.dpi,
        strict_switch_frames=not args.allow_switch_mismatch,
        sample_frames_per_task=args.sample_frames_per_task,
        swap_rb=not args.no_rb_swap,
    )
    print(f"saved summary: {summary_path}")
    for task, info in summary["tasks"].items():
        print(
            f"{task}: episode_{info['episode']} switches={info['switch_frames']} "
            f"curve={info['curve_png']} keyframes={len(info['keyframes'])} "
            f"samples={len(info['sample_frames'])}"
        )


if __name__ == "__main__":
    main()
