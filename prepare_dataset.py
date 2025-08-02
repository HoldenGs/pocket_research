import os
import json
import shutil
from glob import glob
from typing import List

import numpy as np

"""Utility script to prepare the training dataset for imitation_learning.py.

This script performs the following steps:
1. Scans the `data/` directory for recorded sessions
   (folders named `session_*`).
2. For each session it:
   a. Copies the recorded `video.<ext>` file into `data/videos/` and
      renames it to `<start_time>.<ext>` where `start_time` comes from the
      session's `metadata.json`.
   b. This naming convention matches the assumption made in `RCCarDataset`,
      which expects filenames (without extension) to be a floating-point
      UNIX timestamp.
   c. Loads `control_data.json` and appends each record to a global list while
      ensuring the column order is **timestamp, steering, throttle** (note
      that JSON stores *throttle* first).
3. After processing all sessions, it stores the aggregated control signals as a
   NumPy array in `data/control_signals.npy`.

Run this script **once** after collecting new data and before launching
`python imitation_learning.py`. It can safely be re-run – existing videos will
be skipped if they already exist at the destination with the same size.
"""

DATA_DIR = "data"
VIDEO_OUT_DIR = os.path.join(DATA_DIR, "videos")
CONTROL_OUT_NPY = os.path.join(DATA_DIR, "control_signals.npy")


def discover_sessions() -> List[str]:
    """Return list of session directories inside DATA_DIR."""
    pattern = os.path.join(DATA_DIR, "session_*")
    return sorted([d for d in glob(pattern) if os.path.isdir(d)])


def copy_video(session_dir: str, start_time: float) -> str:
    """Copy the session video into the videos directory with the new name.

    Args:
        session_dir: Path to the `session_*` directory.
        start_time: Session start time (UNIX timestamp).

    Returns:
        The destination video path.
    """
    # Detect video file (supports mp4 / avi)
    vid_candidates = [
        f for f in os.listdir(session_dir) if f.startswith("video.")
    ]
    if not vid_candidates:
        raise FileNotFoundError(f"No video file found in {session_dir}")

    video_file = vid_candidates[0]
    src_path = os.path.join(session_dir, video_file)
    ext = os.path.splitext(video_file)[1]
    dst_filename = f"{start_time}{ext}"
    dst_path = os.path.join(VIDEO_OUT_DIR, dst_filename)

    # Only copy if the destination doesn't exist or differs in size
    if (
        not os.path.exists(dst_path)
        or os.path.getsize(dst_path) != os.path.getsize(src_path)
    ):
        print(f"Copying {src_path} -> {dst_path}")
        shutil.copy2(src_path, dst_path)
    else:
        print(
            f"Video already exists, skipping copy: {dst_path}"
        )

    return dst_path


def aggregate_control_data(session_dir: str) -> np.ndarray:
    """Load `control_data.json` and return an array with columns:
    timestamp, steering, throttle (shape: N x 3).
    """
    json_path = os.path.join(session_dir, "control_data.json")
    with open(json_path, "r") as f:
        records = json.load(f)
    data = np.empty((len(records), 3), dtype=np.float64)
    for i, rec in enumerate(records):
        # Column order: timestamp, steering, throttle to match RCCarDataset
        data[i, 0] = rec["timestamp"]
        data[i, 1] = rec["steering"]
        data[i, 2] = rec["throttle"]

    # ------------------------------------------------------------------
    # Trim leading and trailing *idle* periods where both steering and
    # throttle are (near-)zero.  These segments typically correspond to
    # moments before/after the operator takes control and would otherwise
    # bias the model toward predicting zeros.
    # ------------------------------------------------------------------
    idle_mask = (np.abs(data[:, 1]) < 1e-3) & (np.abs(data[:, 2]) < 1e-3)
    # Find first/last indices that are NOT idle
    active_indices = np.where(~idle_mask)[0]
    if active_indices.size == 0:
        # Entire session is idle – return empty array so caller can skip it
        return np.empty((0, 3), dtype=np.float64)

    start_idx, end_idx = active_indices[0], active_indices[-1]
    data = data[start_idx:end_idx + 1]

    return data


def main():
    os.makedirs(VIDEO_OUT_DIR, exist_ok=True)

    sessions = discover_sessions()
    if not sessions:
        print(
            "No session_* directories found inside 'data/'. "
            "Nothing to do."
        )
        return

    all_controls = []
    for session in sessions:
        # Load metadata to get start_time (float)
        meta_path = os.path.join(session, "metadata.json")
        if not os.path.exists(meta_path):
            print(f"Skipping {session}: metadata.json not found")
            continue
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        start_time = metadata.get("start_time")
        if start_time is None:
            print(f"Skipping {session}: start_time not in metadata")
            continue

        # Copy/carry video into videos folder
        copy_video(session, start_time)

        # Aggregate controls
        controls = aggregate_control_data(session)
        all_controls.append(controls)

    if not all_controls:
        print("No control data aggregated – aborting.")
        return

    control_array = np.vstack(all_controls)
    np.save(CONTROL_OUT_NPY, control_array)
    print(
        f"Saved aggregated control data: {CONTROL_OUT_NPY} "
        f"(shape={control_array.shape})"
    )


if __name__ == "__main__":
    main() 