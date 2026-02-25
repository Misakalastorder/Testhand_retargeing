import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


EPS = 1e-8


@dataclass
class RetargetConfig:
	"""
	论文思路下的简化参数：
	1) 通过关键对应点做相似变换配准（R, s, t）
	2) 将每根手指重标定到 Shadow Hand 的近似长度
	3) 从几何关系直接解算 22 个关节角
	"""

	# 对应点：index/middle/ring MCP + wrist
	alignment_human_indices: Tuple[int, int, int, int] = (5, 9, 13, 0)

	# 目标机械手在其基坐标下的参考点（可根据标定再调整）
	alignment_robot_points: Tuple[Tuple[float, float, float], ...] = (
		(0.033, -0.0099, 0.352),
		(0.011, -0.0099, 0.356),
		(-0.011, -0.0099, 0.352),
		(-0.011, -0.005, 0.281),
	)

	# 指骨长度（米），与项目 scaler.py 保持一致
	finger_lengths: Tuple[float, float, float] = (0.045, 0.025, 0.026)
	thumb_lengths: Tuple[float, float, float] = (0.038, 0.032, 0.0275)


def _safe_norm(v: np.ndarray) -> float:
	n = float(np.linalg.norm(v))
	return n if n > EPS else EPS


def _normalize(v: np.ndarray) -> np.ndarray:
	return v / _safe_norm(v)


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
	c = float(np.clip(np.dot(_normalize(v1), _normalize(v2)), -1.0, 1.0))
	return float(np.arccos(c))


def _signed_angle_on_plane(v1: np.ndarray, v2: np.ndarray, n: np.ndarray) -> float:
	n = _normalize(n)
	v1p = v1 - np.dot(v1, n) * n
	v2p = v2 - np.dot(v2, n) * n
	v1p = _normalize(v1p)
	v2p = _normalize(v2p)
	ang = _angle(v1p, v2p)
	sign = float(np.sign(np.dot(n, np.cross(v1p, v2p))))
	return ang * sign


def _clamp_deg(rad_value: float, min_deg: float, max_deg: float) -> float:
	return float(np.clip(rad_value, math.radians(min_deg), math.radians(max_deg)))


def similarity_transform_umeyama(from_pts: np.ndarray, to_pts: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
	"""
	论文中对应点配准的核心数学：
		to ≈ s * R * from + t

	返回:
		R: (3, 3) 旋转矩阵
		s: 标量缩放
		t: (3,) 平移向量
	"""
	x = np.asarray(from_pts, dtype=np.float64)
	y = np.asarray(to_pts, dtype=np.float64)
	if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3:
		raise ValueError(f"相似变换输入必须同形状 (N,3)，收到 x={x.shape}, y={y.shape}")

	n = x.shape[0]
	mx = x.mean(axis=0)
	my = y.mean(axis=0)
	x0 = x - mx
	y0 = y - my

	cov = (y0.T @ x0) / n
	u, sigma, vt = np.linalg.svd(cov)

	sgn = np.eye(3)
	if np.linalg.det(u @ vt) < 0:
		sgn[-1, -1] = -1.0

	r = u @ sgn @ vt
	var_x = float(np.sum(np.var(x, axis=0)))
	scale = float(np.trace(np.diag(sigma) @ sgn) / max(var_x, EPS))
	t = my - scale * (r @ mx)
	return r, scale, t


def apply_similarity(points: np.ndarray, r: np.ndarray, scale: float, t: np.ndarray) -> np.ndarray:
	pts = np.asarray(points, dtype=np.float64)
	return (scale * (r @ pts.T)).T + t


def scale_to_robot_finger_lengths(points21: np.ndarray, cfg: RetargetConfig) -> np.ndarray:
	"""
	按论文“形态归一化”思想，把人手每段指骨重标定到机器人目标长度。
	"""
	p = np.asarray(points21, dtype=np.float64).copy()

	# 每根手指关键点索引（MediaPipe 21 点）
	fingers = [
		((5, 6, 7, 8), cfg.finger_lengths),
		((9, 10, 11, 12), cfg.finger_lengths),
		((13, 14, 15, 16), cfg.finger_lengths),
		((17, 18, 19, 20), cfg.finger_lengths),
		((1, 2, 3, 4), cfg.thumb_lengths),
	]

	for idx_chain, lengths in fingers:
		i0, i1, i2, i3 = idx_chain
		d01 = _normalize(p[i1] - p[i0])
		d12 = _normalize(p[i2] - p[i1])
		d23 = _normalize(p[i3] - p[i2])

		p[i1] = p[i0] + d01 * lengths[0]
		p[i2] = p[i1] + d12 * lengths[1]
		p[i3] = p[i2] + d23 * lengths[2]

	return p


def build_palm_frame(points21: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	用 wrist/index_mcp/pinky_mcp 定义手掌局部坐标系：
		z_palm = normalize((index_mcp-wrist) x (pinky_mcp-wrist))
		x_palm = normalize(middle_mcp-wrist)
		y_palm = z_palm x x_palm
	"""
	p = np.asarray(points21, dtype=np.float64)
	wrist = p[0]
	index_mcp = p[5]
	middle_mcp = p[9]
	pinky_mcp = p[17]

	z_palm = np.cross(index_mcp - wrist, pinky_mcp - wrist)
	if np.linalg.norm(z_palm) < EPS:
		z_palm = np.array([0.0, 0.0, 1.0], dtype=np.float64)
	z_palm = _normalize(z_palm)
	x_palm = _normalize(middle_mcp - wrist)
	y_palm = _normalize(np.cross(z_palm, x_palm))
	return x_palm, y_palm, z_palm


def _compute_finger_angles(points21: np.ndarray, wrist: np.ndarray, mcp: int, pip: int, dip: int, tip: int,
						   x_palm: np.ndarray, z_palm: np.ndarray) -> Tuple[float, float, float, float]:
	proximal = points21[pip] - points21[mcp]
	base = points21[mcp] - wrist
	middle = points21[dip] - points21[pip]
	distal = points21[tip] - points21[dip]

	side = _signed_angle_on_plane(x_palm, proximal, z_palm)
	front = _angle(base, proximal)
	pip_flex = _angle(proximal, middle)
	dip_flex = _angle(middle, distal)
	return side, front, pip_flex, dip_flex


def points_to_shadow_angles(points21: np.ndarray) -> Dict[str, float]:
	"""
	将 21 点人手坐标映射为 22 个 Shadow Hand 关节角（弧度）。
	"""
	p = np.asarray(points21, dtype=np.float64)
	if p.shape != (21, 3):
		raise ValueError(f"期望输入 shape=(21,3)，实际为 {p.shape}")

	wrist = p[0]
	x_palm, _, z_palm = build_palm_frame(p)

	out: Dict[str, float] = {}

	# index/middle/ring
	for prefix, mcp, pip, dip, tip in [
		("I", 5, 6, 7, 8),
		("M", 9, 10, 11, 12),
		("R", 13, 14, 15, 16),
	]:
		side, front, pip_flex, dip_flex = _compute_finger_angles(p, wrist, mcp, pip, dip, tip, x_palm, z_palm)
		out[f"{prefix}MCP_side_joint"] = _clamp_deg(side, -10, 10)
		out[f"{prefix}MCP_front_joint"] = _clamp_deg(front, 0, 100)
		out[f"{prefix}PIP_joint"] = _clamp_deg(pip_flex, 0, 90)
		out[f"{prefix}DIP_joint"] = _clamp_deg(dip_flex, 0, 90)

	# pinky（多一个 metacarpal）
	metacarpal = _angle(p[17] - wrist, p[18] - p[17])
	p_side, p_front, p_pip, p_dip = _compute_finger_angles(p, wrist, 17, 18, 19, 20, x_palm, z_palm)
	out["metacarpal_joint"] = _clamp_deg(metacarpal, 0, 45)
	out["PMCP_side_joint"] = _clamp_deg(p_side, -10, 10)
	out["PMCP_front_joint"] = _clamp_deg(p_front, 0, 100)
	out["PPIP_joint"] = _clamp_deg(p_pip, 0, 90)
	out["PDIP_joint"] = _clamp_deg(p_dip, 0, 90)

	# thumb
	t_base = p[2] - p[1]
	t_mid = p[3] - p[2]
	t_dist = p[4] - p[3]
	t_rot = _signed_angle_on_plane(x_palm, t_base, z_palm)
	t_front = _angle(p[1] - wrist, t_base)
	t_side = math.asin(float(np.clip(np.dot(_normalize(t_mid), z_palm), -1.0, 1.0)))
	t_mid_flex = _angle(t_base, t_mid)
	t_dist_flex = _angle(t_mid, t_dist)

	out["TMCP_rotation_joint"] = _clamp_deg(t_rot, -60, 60)
	out["TMCP_front_joint"] = _clamp_deg(t_front, 0, 70)
	out["TPIP_side_joint"] = _clamp_deg(t_side, -30, 30)
	out["TPIP_front_joint"] = _clamp_deg(t_mid_flex, -12, 12)
	out["TDIP_joint"] = _clamp_deg(t_dist_flex, 0, 90)

	return out


def get_joint_order() -> List[str]:
	return [
		"IMCP_side_joint", "IMCP_front_joint", "IPIP_joint", "IDIP_joint",
		"MMCP_side_joint", "MMCP_front_joint", "MPIP_joint", "MDIP_joint",
		"RMCP_side_joint", "RMCP_front_joint", "RPIP_joint", "RDIP_joint",
		"metacarpal_joint", "PMCP_side_joint", "PMCP_front_joint", "PPIP_joint", "PDIP_joint",
		"TMCP_rotation_joint", "TMCP_front_joint", "TPIP_side_joint", "TPIP_front_joint", "TDIP_joint",
	]


def angles_dict_to_array(angles: Dict[str, float]) -> np.ndarray:
	return np.array([angles[k] for k in get_joint_order()], dtype=np.float64)


def retarget_single_frame(points21_human: np.ndarray, cfg: RetargetConfig = RetargetConfig()) -> Dict[str, float]:
	"""
	单帧输入输出主接口（无仿真）：
	  输入: 人手 21 点 (21,3)
	  输出: Shadow Hand 22 关节角 dict（弧度）

	数学流程:
	  1) 对应点配准（相似变换）
	  2) 统一到机器人手指标度（指骨重标定）
	  3) 几何求角并裁剪到关节范围
	"""
	p = np.asarray(points21_human, dtype=np.float64)
	if p.shape != (21, 3):
		raise ValueError(f"输入必须是 (21,3)，实际为 {p.shape}")

	idx = list(cfg.alignment_human_indices)
	src = p[idx]
	dst = np.asarray(cfg.alignment_robot_points, dtype=np.float64)

	r, s, t = similarity_transform_umeyama(src, dst)
	aligned = apply_similarity(p, r, s, t)
	scaled = scale_to_robot_finger_lengths(aligned, cfg)
	return points_to_shadow_angles(scaled)


def retarget_single_frame_array(points21_human: np.ndarray, cfg: RetargetConfig = RetargetConfig()) -> np.ndarray:
	angles = retarget_single_frame(points21_human, cfg)
	return angles_dict_to_array(angles)


if __name__ == "__main__":
	# 最小示例：随机输入只用于演示接口（真实使用请替换为真实手部关键点）
	demo = np.zeros((21, 3), dtype=np.float64)
	# 给出一个简单张手姿态，避免退化
	demo[0] = np.array([0.0, 0.0, 0.0])
	demo[5] = np.array([0.03, 0.0, 0.0])
	demo[9] = np.array([0.00, 0.02, 0.0])
	demo[13] = np.array([-0.03, 0.0, 0.0])
	demo[17] = np.array([-0.05, -0.01, 0.0])
	demo[1:5] = np.array([[0.02, -0.01, 0.0], [0.04, -0.015, 0.0], [0.06, -0.020, 0.0], [0.08, -0.025, 0.0]])
	demo[6:9] = np.array([[0.05, 0.02, 0.0], [0.07, 0.03, 0.0], [0.09, 0.035, 0.0]])
	demo[10:13] = np.array([[0.00, 0.04, 0.0], [0.00, 0.06, 0.0], [0.00, 0.08, 0.0]])
	demo[14:17] = np.array([[-0.03, 0.03, 0.0], [-0.05, 0.04, 0.0], [-0.07, 0.045, 0.0]])
	demo[18:21] = np.array([[-0.07, 0.00, 0.0], [-0.09, 0.005, 0.0], [-0.11, 0.008, 0.0]])

	result = retarget_single_frame(demo)
	for k in get_joint_order():
		print(f"{k}: {result[k]:.6f}")
