"""
Convert X-One ARIO data from BOS to LeRobot v2.1 format for pi0.5 fine-tuning.

Source: bos://world-model-data/ARIO-format/xspark/xone26/
Target: bos://world-model-data/rongyinze/dataset/x-one/lerobotDataset

Each episode in the source contains:
  - qpos.pt: [T, 14] joint positions (state). 14 = left_joint(6)+left_gripper(1)+right_joint(6)+right_gripper(1)
  - raw_video/cam_high.mp4, cam_left_wrist.mp4, cam_right_wrist.mp4: 640x480 @ 30fps
  - instructions.json: task-level language instruction

For ALOHA-style training, action[t] = qpos[t+1] (next-frame joint target).

Usage:
  uv run scripts/convert_xone_to_lerobot.py
  uv run scripts/convert_xone_to_lerobot.py --max_episodes 10   # quick test
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.constants import HF_LEROBOT_HOME
from megfile import smart_listdir, smart_sync

BOS_SRC = "bos://world-model-data/ARIO-format/xspark/xone26/"
BOS_DST = "bos://world-model-data/rongyinze/dataset/x-one/lerobotDataset"
REPO_ID = "xone/xone26"
FPS = 30

MOTORS = [
    "left_waist", "left_shoulder", "left_elbow",
    "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate",
    "left_gripper",
    "right_waist", "right_shoulder", "right_elbow",
    "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate",
    "right_gripper",
]

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def create_empty_dataset(dataset_dir: Path) -> LeRobotDataset:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(MOTORS),),
            "names": [MOTORS],
        },
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    return LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=dataset_dir,
        robot_type="aloha",
        features=features,
        use_videos=True,
        image_writer_processes=4,
        image_writer_threads=4,
    )


def decode_video_frames(video_path: str) -> np.ndarray:
    """Read all frames from a video file. Returns [T, H, W, 3] uint8 RGB."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def load_episode_from_bos(task_name: str, episode_name: str, tmp_dir: str) -> dict:
    """Download one episode from BOS and load its data."""
    bos_ep = f"{BOS_SRC}{task_name}/{episode_name}/"
    local_ep = f"{tmp_dir}/{episode_name}/"
    smart_sync(bos_ep, local_ep)

    qpos = torch.load(f"{local_ep}/qpos.pt", map_location="cpu", weights_only=False)

    with open(f"{local_ep}/instructions.json") as f:
        instructions = json.load(f)
    task_instruction = task_name
    if instructions.get("sub_instructions"):
        all_instr = [s["instruction"] for s in instructions["sub_instructions"] if s.get("instruction")]
        if all_instr:
            task_instruction = " ".join(all_instr)

    videos = {}
    for cam in CAMERAS:
        video_path = f"{local_ep}/raw_video/{cam}.mp4"
        videos[cam] = decode_video_frames(video_path)

    return {
        "qpos": qpos,
        "videos": videos,
        "task": task_instruction,
    }


def populate_episode(dataset: LeRobotDataset, episode_data: dict):
    """Add one episode's frames to the dataset."""
    qpos = episode_data["qpos"]  # [T, 14]
    videos = episode_data["videos"]  # {cam: [T, H, W, 3]}
    task = episode_data["task"]

    T = qpos.shape[0]
    cam_frames = {cam: videos[cam] for cam in CAMERAS}

    for cam in CAMERAS:
        n_vid = cam_frames[cam].shape[0]
        if n_vid != T:
            print(f"  Warning: {cam} has {n_vid} frames but qpos has {T}, truncating to min")
            T = min(T, n_vid)

    # action[t] = qpos[t+1] for t < T-1; action[T-1] = qpos[T-1] (repeat last)
    actions = torch.zeros_like(qpos)
    actions[:-1] = qpos[1:]
    actions[-1] = qpos[-1]

    for t in range(T):
        frame = {
            "observation.state": qpos[t].numpy(),
            "action": actions[t].numpy(),
            "task": task,
        }
        for cam in CAMERAS:
            frame[f"observation.images.{cam}"] = cam_frames[cam][t]

        dataset.add_frame(frame)

    dataset.save_episode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes per task (for testing)")
    parser.add_argument("--output_dir", type=str, default=None, help="Root dir for LeRobot datasets (dataset stored under <output_dir>/xone/xone26/)")
    args = parser.parse_args()

    output_root = Path(args.output_dir) if args.output_dir else HF_LEROBOT_HOME / REPO_ID

    print(f"Dataset will be written to: {output_root}")

    tasks = smart_listdir(BOS_SRC)
    print(f"Found {len(tasks)} tasks: {tasks}")

    dataset = create_empty_dataset(output_root)

    for task_name in sorted(tasks):
        episodes = sorted(smart_listdir(f"{BOS_SRC}{task_name}/"))
        if args.max_episodes:
            episodes = episodes[: args.max_episodes]
        print(f"\nTask: {task_name} ({len(episodes)} episodes)")

        for ep_idx, ep_name in enumerate(episodes):
            with tempfile.TemporaryDirectory() as tmp_dir:
                print(f"  [{ep_idx+1}/{len(episodes)}] {ep_name} ...", end=" ", flush=True)
                try:
                    episode_data = load_episode_from_bos(task_name, ep_name, tmp_dir)
                    populate_episode(dataset, episode_data)
                    print(f"ok (T={episode_data['qpos'].shape[0]})")
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

    print(f"\nDataset saved to: {output_root}")
    print(f"Total episodes: {dataset.meta.total_episodes}, total frames: {dataset.meta.total_frames}")

    # Upload to BOS
    print(f"\nUploading to {BOS_DST} ...")
    smart_sync(str(output_root) + "/", BOS_DST + "/")
    print("Done!")


if __name__ == "__main__":
    main()
