import argparse
import json
import os

import cv2
import h5py
import numpy as np


DEFAULT_MODE_NAMES = (
    "stabilize",
    "absolute",
    "relative_translation_close",
    "relative_translation_apart",
    "relative_rotation",
    "mixed",
)
DEFAULT_RATIO_LABELS = ("absolute", "relative_translation", "relative_rotation")
MODE_COLORS = {
    "stabilize": (170, 170, 170),
    "absolute": (80, 170, 255),
    "relative_translation": (70, 210, 120),
    "relative_translation_close": (70, 210, 120),
    "relative_translation_apart": (80, 190, 255),
    "relative_rotation": (230, 120, 230),
    "mixed": (80, 200, 240),
}


def decode_image(encoded_image):
    image = cv2.imdecode(np.frombuffer(encoded_image, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode JPEG image from HDF5.")
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def read_json_attr(attrs, name, default):
    if name not in attrs:
        return default
    value = attrs[name]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return tuple(json.loads(value))


def color_for_mode(mode, mode_names=None):
    mode = int(mode)
    if mode_names is not None and 0 <= mode < len(mode_names):
        return MODE_COLORS.get(mode_names[mode], (255, 255, 255))
    if 0 <= mode < len(DEFAULT_MODE_NAMES):
        return MODE_COLORS.get(DEFAULT_MODE_NAMES[mode], (255, 255, 255))
    return (255, 255, 255)


def draw_ratio_bars(panel, ratio, ratio_mask, labels, start_x, start_y, width=360, bar_h=22):
    colors = ((80, 170, 255), (70, 210, 120), (230, 120, 230))
    for idx, (label, value) in enumerate(zip(labels, ratio)):
        y = start_y + idx * 36
        cv2.putText(panel, f"{label}: {value:.3f}", (start_x, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (245, 245, 245), 1, cv2.LINE_AA)
        x0 = start_x + 210
        cv2.rectangle(panel, (x0, y), (x0 + width, y + bar_h), (70, 70, 70), 1)
        fill_w = int(width * float(np.clip(value, 0.0, 1.0))) if ratio_mask > 0.0 else 0
        cv2.rectangle(panel, (x0, y), (x0 + fill_w, y + bar_h), colors[idx], -1)


def draw_mode_timeline(panel, modes, frame_idx, mode_names, y, x0=24, height=16):
    width = panel.shape[1] - 2 * x0
    num_frames = len(modes)
    if num_frames == 0:
        return
    for idx, mode in enumerate(modes):
        x = x0 + int(width * idx / num_frames)
        x_next = x0 + int(width * (idx + 1) / num_frames)
        cv2.rectangle(panel, (x, y), (max(x_next, x + 1), y + height), color_for_mode(mode, mode_names), -1)
    cursor_x = x0 + int(width * frame_idx / max(num_frames - 1, 1))
    cv2.rectangle(panel, (cursor_x - 2, y - 4), (cursor_x + 2, y + height + 4), (255, 255, 255), -1)
    cv2.putText(panel, "mode timeline", (x0, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
                cv2.LINE_AA)
    legend_y = y + height + 24
    legend_cols = min(3, max(len(mode_names), 1))
    legend_col_width = (panel.shape[1] - 2 * x0) // legend_cols
    for mode_id, name in enumerate(mode_names):
        legend_x = x0 + (mode_id % legend_cols) * legend_col_width
        legend_row_y = legend_y + (mode_id // legend_cols) * 20
        cv2.rectangle(
            panel,
            (legend_x, legend_row_y - 12),
            (legend_x + 14, legend_row_y + 2),
            color_for_mode(mode_id, mode_names),
            -1,
        )
        cv2.putText(panel, f"{mode_id}:{name}", (legend_x + 20, legend_row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220, 220, 220), 1, cv2.LINE_AA)


def make_frame(h5, frame_idx, mode_names, ratio_labels, title, camera_size):
    images = []
    for camera_name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        image = decode_image(h5[f"/observations/images/{camera_name}"][frame_idx])
        image = cv2.resize(image, camera_size)
        cv2.putText(image, camera_name, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(image, camera_name, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        images.append(image)
    image_row = cv2.hconcat(images)

    mode = int(h5["/mode"][frame_idx])
    ratio = np.asarray(h5["/ratio"][frame_idx], dtype=np.float32)
    ratio_mask = float(h5["/ratio_mask"][frame_idx])
    motion_energy = np.asarray(h5["/motion_energy"][frame_idx], dtype=np.float32)
    chunk_horizon = int(h5.attrs.get("chunk_horizon", -1))
    mode_name = mode_names[mode] if 0 <= mode < len(mode_names) else f"unknown_{mode}"

    panel = np.full((290, image_row.shape[1], 3), 28, dtype=np.uint8)
    mode_color = color_for_mode(mode, mode_names)
    cv2.putText(panel, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(panel, f"frame={frame_idx:04d}  chunk_horizon={chunk_horizon}", (24, 64), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, f"mode={mode}:{mode_name}", (24, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2,
                cv2.LINE_AA)
    cv2.putText(panel, f"ratio_mask={ratio_mask:.1f}", (24, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, "motion_energy=[" + ", ".join(f"{v:.4f}" for v in motion_energy) + "]", (24, 162),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    draw_ratio_bars(panel, ratio, ratio_mask, ratio_labels, image_row.shape[1] // 2, 54)
    draw_mode_timeline(panel, h5["/mode"][:], frame_idx, mode_names, 204)
    return cv2.vconcat([panel, image_row])


def visualize_episode(episode_hdf5, output_path, fps, camera_width, camera_height, title):
    with h5py.File(episode_hdf5, "r") as h5:
        mode_names = read_json_attr(h5.attrs, "mode_names", DEFAULT_MODE_NAMES)
        ratio_labels = read_json_attr(h5.attrs, "ratio_labels", DEFAULT_RATIO_LABELS)
        num_frames = h5["/mode"].shape[0]
        if num_frames == 0:
            raise ValueError(f"No frames to visualize in {episode_hdf5}")
        first_frame = make_frame(h5, 0, mode_names, ratio_labels, title, (camera_width, camera_height))
        height, width = first_frame.shape[:2]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_path}")
        writer.write(first_frame)
        for frame_idx in range(1, num_frames):
            writer.write(make_frame(h5, frame_idx, mode_names, ratio_labels, title, (camera_width, camera_height)))
        writer.release()


def main():
    parser = argparse.ArgumentParser(description="Visualize mode/ratio auxiliary labels over processed pi0 HDF5 data.")
    parser.add_argument("episode_hdf5", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--camera-width", type=int, default=320)
    parser.add_argument("--camera-height", type=int, default=240)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    title = args.title or os.path.basename(os.path.dirname(args.episode_hdf5))
    visualize_episode(args.episode_hdf5, args.output_path, args.fps, args.camera_width, args.camera_height, title)
    print(f"saved video: {args.output_path}")


if __name__ == "__main__":
    main()
