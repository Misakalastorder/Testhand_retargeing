import os
import h5py
import numpy as np


# h5 文件路径
H5_PATH = r"D:\\2026\\code\\others\\Testhand_retargeing\\hand_pose_retargeting-main\\data\\test_glove_data0204_aligned.h5"

# 如果你知道具体的数据集名字，直接写在这里。
# 不知道的话先跑一遍脚本，看下面打印出来的名字，再改这个变量。
DATASET_NAME = "测试-ceshi/r_glove_pos"  # 例如 "glove_data" / "data" 等


def list_h5_contents(h5_path: str) -> None:
    """打印 h5 文件中的所有对象名称和类型。"""
    if not os.path.exists(h5_path):
        print(f"h5 文件不存在: {h5_path}")
        return

    with h5py.File(h5_path, "r") as f:
        print(f"打开 h5 文件: {h5_path}")
        print("文件中包含的对象:")
        def _print_name(name, obj):
            print(f"  - {name}: type={type(obj)}")
        f.visititems(_print_name)


def load_hand_data(h5_path: str, dataset_name: str) -> np.ndarray:
    """
    从 h5 中读取 (frames, 25, 3) 的人手数据。
    返回: numpy 数组，形状应为 (frames, 25, 3)
    """
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"h5 文件不存在: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if dataset_name not in f:
            raise KeyError(
                f"数据集 '{dataset_name}' 不在文件中，请先运行 list_h5_contents() 查看实际名字"
            )
        data = f[dataset_name][...]
        print(f"读取数据集 '{dataset_name}', 形状: {data.shape}")
        return data


def main():
    # 1. 先列出 h5 内容，确定数据集名字
    list_h5_contents(H5_PATH)

    # 2. 如果已经确认 DATASET_NAME 正确，可以取消下面两行的注释读取数据
    hand_data = load_hand_data(H5_PATH, DATASET_NAME)
    print("hand_data 形状:", hand_data.shape)  # 期望是 (frames, 25, 3)

    # 3. TODO: 在这里接入“人手 → 灵巧手关节”的重定向逻辑，
    #    比如循环每一帧 hand_data[frame]，调用你的重定向函数，
    #    然后把输出保存到一个新的文件（.npy / .h5 / .txt 等）。


if __name__ == "__main__":
    main()