import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


DEFAULT_DATASET_DIRS = (
    "training_data_realworld_carry_basket",
    "training_data_realworld_dual_pour_water",
    "training_data_realworld_dual_sample_loading",
)
ARM_NAMES = ("left", "right")
BACKUP_DATASET_NAME = "ee_pose_rpy_wrapped_original"
REPORT_NAME = "rpy_wrap_repair_report.json"


def numeric_sort_key(path: Path):
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.name)


def unwrap_period(values, period, threshold):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {values.shape}.")
    if len(values) == 0:
        return values.copy()

    unwrapped = np.empty_like(values, dtype=np.float64)
    offset = np.zeros(values.shape[1], dtype=np.float64)
    unwrapped[0] = values[0]

    for i in range(1, len(values)):
        candidate = values[i] + offset
        delta = candidate - unwrapped[i - 1]
        for axis in range(values.shape[1]):
            while delta[axis] > threshold:
                offset[axis] -= period
                delta[axis] -= period
            while delta[axis] < -threshold:
                offset[axis] += period
                delta[axis] += period
        unwrapped[i] = values[i] + offset

    return unwrapped


def axis_jump_count(rpy, threshold):
    if len(rpy) < 2:
        return 0
    return int(np.sum(np.abs(np.diff(rpy, axis=0)) > threshold))


def frame_jump_count(rpy, threshold):
    if len(rpy) < 2:
        return 0
    return int(np.sum(np.any(np.abs(np.diff(rpy, axis=0)) > threshold, axis=1)))


def max_step_abs(rpy):
    if len(rpy) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(rpy, axis=0))))


def summarize_repair(original_pose, repaired_pose, threshold):
    original_rpy = original_pose[:, 3:6]
    repaired_rpy = repaired_pose[:, 3:6]
    correction = repaired_rpy - original_rpy
    nonzero_correction = np.any(np.abs(correction) > 1e-12, axis=1)

    return {
        "frames": int(original_pose.shape[0]),
        "raw_axis_jumps": axis_jump_count(original_rpy, threshold),
        "raw_frame_jumps": frame_jump_count(original_rpy, threshold),
        "repaired_axis_jumps": axis_jump_count(repaired_rpy, threshold),
        "repaired_frame_jumps": frame_jump_count(repaired_rpy, threshold),
        "raw_max_step_abs": max_step_abs(original_rpy),
        "repaired_max_step_abs": max_step_abs(repaired_rpy),
        "corrected_frames": int(np.sum(nonzero_correction)),
        "max_abs_correction": float(np.max(np.abs(correction))) if correction.size else 0.0,
        "raw_min": original_rpy.min(axis=0).tolist(),
        "raw_max": original_rpy.max(axis=0).tolist(),
        "repaired_min": repaired_rpy.min(axis=0).tolist(),
        "repaired_max": repaired_rpy.max(axis=0).tolist(),
    }


def repair_pose_array(ee_pose, period, threshold):
    repaired = np.asarray(ee_pose).copy()
    repaired[:, 3:6] = unwrap_period(repaired[:, 3:6], period=period, threshold=threshold)
    return repaired


def get_source_pose(group, use_backup_source):
    pose = group["ee_pose"][()]
    if use_backup_source and BACKUP_DATASET_NAME in group:
        return group[BACKUP_DATASET_NAME][()]
    return pose


def ensure_backup(group, current_pose, overwrite_backup):
    if BACKUP_DATASET_NAME in group:
        if overwrite_backup:
            del group[BACKUP_DATASET_NAME]
        else:
            return False
    group.create_dataset(BACKUP_DATASET_NAME, data=current_pose)
    return True


def write_repair_attrs(dataset, period, threshold):
    now = datetime.now(timezone.utc).isoformat()
    dataset.attrs["rpy_wrap_repaired"] = True
    dataset.attrs["rpy_wrap_period"] = float(period)
    dataset.attrs["rpy_wrap_threshold"] = float(threshold)
    dataset.attrs["rpy_wrap_backup_dataset"] = BACKUP_DATASET_NAME
    dataset.attrs["rpy_wrap_repaired_at"] = now


def delete_repair_attrs(dataset):
    for key in (
        "rpy_wrap_repaired",
        "rpy_wrap_period",
        "rpy_wrap_threshold",
        "rpy_wrap_backup_dataset",
        "rpy_wrap_repaired_at",
    ):
        if key in dataset.attrs:
            del dataset.attrs[key]


def repair_episode_file(
    hdf5_path,
    period,
    threshold,
    arms,
    in_place,
    restore,
    use_backup_source,
    overwrite_backup,
):
    file_report = {"file": hdf5_path.name, "arms": {}}
    mode = "r+" if in_place else "r"

    with h5py.File(hdf5_path, mode) as root:
        for arm in arms:
            group_key = f"/slave_{arm}_arm"
            if group_key not in root:
                raise KeyError(f"{hdf5_path} does not contain {group_key}.")
            group = root[group_key]
            if "ee_pose" not in group:
                raise KeyError(f"{hdf5_path} does not contain {group_key}/ee_pose.")

            dataset = group["ee_pose"]
            current_pose = dataset[()]

            if restore:
                if BACKUP_DATASET_NAME not in group:
                    raise KeyError(f"{hdf5_path}:{group_key} has no {BACKUP_DATASET_NAME} backup.")
                backup_pose = group[BACKUP_DATASET_NAME][()]
                if backup_pose.shape != current_pose.shape:
                    raise ValueError(
                        f"{hdf5_path}:{group_key} backup shape {backup_pose.shape} "
                        f"does not match ee_pose shape {current_pose.shape}."
                    )
                summary = summarize_repair(current_pose, backup_pose, threshold)
                if in_place:
                    dataset[...] = backup_pose.astype(dataset.dtype, copy=False)
                    delete_repair_attrs(dataset)
                summary["restored_from_backup"] = bool(in_place)
                file_report["arms"][arm] = summary
                continue

            source_pose = get_source_pose(group, use_backup_source)
            if source_pose.shape != current_pose.shape:
                raise ValueError(
                    f"{hdf5_path}:{group_key} source shape {source_pose.shape} "
                    f"does not match ee_pose shape {current_pose.shape}."
                )
            repaired_pose = repair_pose_array(source_pose, period=period, threshold=threshold)
            summary = summarize_repair(source_pose, repaired_pose, threshold)

            if in_place:
                backup_created = ensure_backup(group, current_pose, overwrite_backup=overwrite_backup)
                dataset[...] = repaired_pose.astype(dataset.dtype, copy=False)
                write_repair_attrs(dataset, period=period, threshold=threshold)
                summary["backup_created"] = backup_created
                summary["wrote_repair"] = True
            else:
                summary["backup_created"] = False
                summary["wrote_repair"] = False
            file_report["arms"][arm] = summary

    return file_report


def aggregate_reports(file_reports, arms):
    total = {
        "files": len(file_reports),
        "arms": {
            arm: {
                "raw_axis_jumps": 0,
                "raw_frame_jumps": 0,
                "repaired_axis_jumps": 0,
                "repaired_frame_jumps": 0,
                "corrected_frames": 0,
                "max_raw_step_abs": 0.0,
                "max_repaired_step_abs": 0.0,
                "max_abs_correction": 0.0,
            }
            for arm in arms
        },
    }
    for report in file_reports:
        for arm in arms:
            arm_report = report["arms"][arm]
            total_arm = total["arms"][arm]
            total_arm["raw_axis_jumps"] += arm_report["raw_axis_jumps"]
            total_arm["raw_frame_jumps"] += arm_report["raw_frame_jumps"]
            total_arm["repaired_axis_jumps"] += arm_report["repaired_axis_jumps"]
            total_arm["repaired_frame_jumps"] += arm_report["repaired_frame_jumps"]
            total_arm["corrected_frames"] += arm_report["corrected_frames"]
            total_arm["max_raw_step_abs"] = max(total_arm["max_raw_step_abs"], arm_report["raw_max_step_abs"])
            total_arm["max_repaired_step_abs"] = max(
                total_arm["max_repaired_step_abs"], arm_report["repaired_max_step_abs"]
            )
            total_arm["max_abs_correction"] = max(total_arm["max_abs_correction"], arm_report["max_abs_correction"])
    return total


def process_dataset_dir(dataset_dir, args, arms):
    hdf5_files = sorted(dataset_dir.glob("*.hdf5"), key=numeric_sort_key)
    if args.num_episodes is not None:
        hdf5_files = hdf5_files[: args.num_episodes]
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {dataset_dir}")

    reports = []
    for hdf5_path in hdf5_files:
        report = repair_episode_file(
            hdf5_path,
            period=args.period,
            threshold=args.threshold,
            arms=arms,
            in_place=args.in_place,
            restore=args.restore,
            use_backup_source=not args.no_use_backup_source,
            overwrite_backup=args.overwrite_backup,
        )
        reports.append(report)

    dataset_report = {
        "dataset_dir": str(dataset_dir),
        "mode": "restore" if args.restore else "repair",
        "wrote_changes": bool(args.in_place),
        "period": float(args.period),
        "threshold": float(args.threshold),
        "arms": list(arms),
        "summary": aggregate_reports(reports, arms),
        "files": reports,
    }

    if args.in_place and not args.no_report:
        report_path = dataset_dir / REPORT_NAME
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(dataset_report, f, indent=2)

    return dataset_report


def print_summary(dataset_report):
    action = "restored" if dataset_report["mode"] == "restore" else "repaired"
    write_note = "wrote" if dataset_report["wrote_changes"] else "dry-run"
    print(f"{dataset_report['dataset_dir']} [{action}, {write_note}]")
    for arm, summary in dataset_report["summary"]["arms"].items():
        print(
            "  "
            f"{arm}: raw_axis_jumps={summary['raw_axis_jumps']} -> "
            f"repaired_axis_jumps={summary['repaired_axis_jumps']}, "
            f"corrected_frames={summary['corrected_frames']}, "
            f"max_step={summary['max_raw_step_abs']:.6f} -> {summary['max_repaired_step_abs']:.6f}, "
            f"max_abs_correction={summary['max_abs_correction']:.6f}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Repair real-world ee_pose RPY wrap jumps. The problematic datasets wrap R/P/Y into roughly "
            "[-0.18, 0.18], so this script unwraps each RPY axis with period 0.36."
        )
    )
    parser.add_argument(
        "dataset_dirs",
        nargs="*",
        type=Path,
        default=[Path(name) for name in DEFAULT_DATASET_DIRS],
        help="Dataset directories containing flat *.hdf5 episode files.",
    )
    parser.add_argument("--period", type=float, default=0.36, help="Wrap period for each RPY axis.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Step threshold for unwrap. Defaults to period / 2.",
    )
    parser.add_argument("--left-only", action="store_true", help="Repair only the left arm.")
    parser.add_argument("--right-only", action="store_true", help="Repair only the right arm.")
    parser.add_argument("--num-episodes", type=int, default=None, help="Debug option: process only first N episodes.")
    parser.add_argument("--in-place", action="store_true", help="Write repaired RPY values back into ee_pose.")
    parser.add_argument("--restore", action="store_true", help="Restore ee_pose from ee_pose_rpy_wrapped_original.")
    parser.add_argument(
        "--no-use-backup-source",
        action="store_true",
        help="When repairing a file that already has a backup, use current ee_pose instead of the backup.",
    )
    parser.add_argument(
        "--overwrite-backup",
        action="store_true",
        help="Replace an existing backup with the current ee_pose before writing the repair.",
    )
    parser.add_argument("--no-report", action="store_true", help=f"Do not write {REPORT_NAME}.")
    return parser


def main():
    args = build_parser().parse_args()
    if args.threshold is None:
        args.threshold = args.period / 2.0
    if args.period <= 0.0:
        raise ValueError("--period must be positive.")
    if args.threshold <= 0.0:
        raise ValueError("--threshold must be positive.")
    if args.threshold > args.period:
        raise ValueError("--threshold must be no larger than --period.")
    if args.left_only and args.right_only:
        raise ValueError("--left-only and --right-only cannot both be set.")
    if args.restore and args.overwrite_backup:
        raise ValueError("--restore and --overwrite-backup cannot both be set.")

    if args.left_only:
        arms = ("left",)
    elif args.right_only:
        arms = ("right",)
    else:
        arms = ARM_NAMES

    for dataset_dir in args.dataset_dirs:
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
        report = process_dataset_dir(dataset_dir, args, arms)
        print_summary(report)

    if not args.in_place:
        print("Dry run only. Pass --in-place to write repaired ee_pose values and backups.")


if __name__ == "__main__":
    main()
