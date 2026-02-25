import argparse
import csv
import os
from typing import List

import h5py
import numpy as np

from algorithm import angles_dict_to_array, get_joint_order, retarget_single_frame
# D:\2026\code\test_other\Testhand_retargeing\hand_pose_retargeting-main\pose_retargeting\optimization\model\shadowrobot.urdf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_H5_PATH = os.path.join(ROOT_DIR, "data", "test_glove_data0204_aligned.h5")
DEFAULT_DATASET_NAME = "测试-ceshi/r_glove_pos"

def list_h5_contents(h5_path: str) -> None:
    if not os.path.exists(h5_path):
        print(f"h5 文件不存在: {h5_path}")
        return

    with h5py.File(h5_path, "r") as h5_file:
        print(f"打开 h5 文件: {h5_path}")
        print("文件中包含的对象:")

        def _print_name(name, obj):
            print(f"  - {name}: type={type(obj)}")

        h5_file.visititems(_print_name)


def load_hand_data(h5_path: str, dataset_name: str) -> np.ndarray:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"h5 文件不存在: {h5_path}")

    with h5py.File(h5_path, "r") as h5_file:
        if dataset_name not in h5_file:
            raise KeyError(f"数据集 '{dataset_name}' 不在文件中，请先用 --list_h5 查看")
        data = h5_file[dataset_name][...]

    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"期望输入形状为 (frames, joints, 3)，实际为 {data.shape}")
    if data.shape[1] < 21:
        raise ValueError(f"至少需要 21 个关键点，实际为 {data.shape[1]}")

    print(f"读取数据集 '{dataset_name}', 形状: {data.shape}")
    return data


def save_angles(output_prefix: str, joint_names: List[str], angles: np.ndarray) -> None:
    npy_path = output_prefix + ".npy"
    csv_path = output_prefix + ".csv"

    np.save(npy_path, angles)

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame"] + joint_names)
        for frame_idx in range(angles.shape[0]):
            writer.writerow([frame_idx] + angles[frame_idx].tolist())

    print(f"关节角已保存: {npy_path}")
    print(f"关节角已保存: {csv_path}")


def save_angles_h5(output_h5_path: str, joint_names: List[str], angles: np.ndarray,
                   dataset_name: str = "outputs") -> None:
    with h5py.File(output_h5_path, "w") as h5_file:
        h5_file.create_dataset(dataset_name, data=angles)
        name_array = np.array(joint_names, dtype="S")
        h5_file.create_dataset("joint_names", data=name_array)
    print(f"关节角已保存: {output_h5_path} -> {dataset_name}, shape={angles.shape}")


def retarget_all_frames(hand_data: np.ndarray) -> np.ndarray:
    if hand_data.ndim != 3 or hand_data.shape[1] < 21 or hand_data.shape[2] != 3:
        raise ValueError(f"输入应为 (frames, joints, 3) 且 joints>=21，实际为 {hand_data.shape}")

    all_angles = []
    for frame_index in range(hand_data.shape[0]):
        frame_points = np.asarray(hand_data[frame_index][:21], dtype=np.float64)
        angles_dict = retarget_single_frame(frame_points)
        angles_rad = angles_dict_to_array(angles_dict)
        all_angles.append(angles_rad)
    return np.asarray(all_angles, dtype=np.float64)


def run_all_frames_to_h5(h5_path: str, dataset_name: str, output_h5_path: str,
                         output_dataset_name: str = "outputs") -> None:
    hand_data = load_hand_data(h5_path, dataset_name)
    joint_names = get_joint_order()
    all_angles = retarget_all_frames(hand_data)
    save_angles_h5(output_h5_path, joint_names, all_angles, dataset_name=output_dataset_name)


def run_simple_single_frame(
    h5_path: str,
    dataset_name: str,
    frame_index: int,
    output_prefix: str,
) -> None:
    hand_data = load_hand_data(h5_path, dataset_name)
    if not (0 <= frame_index < hand_data.shape[0]):
        raise IndexError(f"frame_index 越界: {frame_index}, 可用范围 [0, {hand_data.shape[0] - 1}]")

    frame_points = np.asarray(hand_data[frame_index][:21], dtype=np.float64)
    angles_dict = retarget_single_frame(frame_points)
    joint_names = get_joint_order()
    angles_rad = angles_dict_to_array(angles_dict)
    angles_deg = np.rad2deg(angles_rad)

    print(f"单帧重定向完成: frame={frame_index}")
    print("关节角(弧度):")
    for name, value in zip(joint_names, angles_rad):
        print(f"  {name}: {value:.6f}")

    print("关节角(角度):")
    for name, value in zip(joint_names, angles_deg):
        print(f"  {name}: {value:.3f}")

    single_frame_output = angles_rad.reshape(1, -1)
    save_angles(output_prefix, joint_names, single_frame_output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线手势重定向并输出灵巧手关节角")
    parser.add_argument("--h5_path", type=str, default=DEFAULT_H5_PATH, help="输入 h5 文件路径")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_NAME, help="h5 数据集路径")
    parser.add_argument("--output_prefix", type=str, default=os.path.join(ROOT_DIR, "data", "dexterous_hand_angles_linker"),
                        help="输出文件前缀（会生成 .npy 和 .csv）")
    parser.add_argument("--list_h5", action="store_true", default=False, help="只列出 h5 内容，不执行重定向")
    parser.add_argument("--frame_index", type=int, default=0, help="单帧模式读取的帧下标")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_h5:
        list_h5_contents(args.h5_path)
        return

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    run_simple_single_frame(
        h5_path=args.h5_path,
        dataset_name=args.dataset,
        frame_index=args.frame_index,
        output_prefix=args.output_prefix,
    )

    run_all_frames_to_h5(
        h5_path=args.h5_path,
        dataset_name=args.dataset,
        output_h5_path=args.output_prefix + ".h5",
    )

if __name__ == "__main__":
    main()