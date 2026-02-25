import math
from typing import Dict, List

import numpy as np


def _safe_norm(vec: np.ndarray, eps: float = 1e-8) -> float:
    value = float(np.linalg.norm(vec))
    return value if value > eps else eps


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / _safe_norm(vec)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = _normalize(v1)
    n2 = _normalize(v2)
    cosine = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    return float(np.arccos(cosine))


def _signed_angle_on_plane(v1: np.ndarray, v2: np.ndarray, normal: np.ndarray) -> float:
    normal = _normalize(normal)
    v1_proj = v1 - np.dot(v1, normal) * normal
    v2_proj = v2 - np.dot(v2, normal) * normal
    v1_proj = _normalize(v1_proj)
    v2_proj = _normalize(v2_proj)
    unsigned = _angle_between(v1_proj, v2_proj)
    sign = np.sign(np.dot(normal, np.cross(v1_proj, v2_proj)))
    return float(unsigned * sign)


def _clamp_rad(angle: float, min_deg: float, max_deg: float) -> float:
    min_rad = math.radians(min_deg)
    max_rad = math.radians(max_deg)
    return float(np.clip(angle, min_rad, max_rad))


def _finger_flex(points: np.ndarray, a: int, b: int, c: int) -> float:
    return _angle_between(points[b] - points[a], points[c] - points[b])


def retarget_single_frame(points_21x3: np.ndarray) -> Dict[str, float]:
    """
    输入:
        points_21x3: numpy.ndarray, shape=(21, 3)
            关键点顺序默认符合 MediaPipe Hands (0~20)
    输出:
        dict[str, float]: 22 个 Shadow Hand 关节角 (单位: 弧度)
    """
    points = np.asarray(points_21x3, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"期望输入 shape=(21, 3)，实际为 {points.shape}")

    wrist = points[0]
    palm_ref_a = points[5] - wrist
    palm_ref_b = points[17] - wrist
    palm_normal = np.cross(palm_ref_a, palm_ref_b)
    if np.linalg.norm(palm_normal) < 1e-8:
        palm_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    palm_normal = _normalize(palm_normal)

    palm_forward = points[9] - wrist
    if np.linalg.norm(palm_forward) < 1e-8:
        palm_forward = points[5] - wrist
    palm_forward = _normalize(palm_forward)

    angles: Dict[str, float] = {}

    def add_standard_finger(prefix: str, mcp: int, pip: int, dip: int, tip: int) -> None:
        mcp_to_pip = points[pip] - points[mcp]
        wrist_to_mcp = points[mcp] - wrist

        side = _signed_angle_on_plane(palm_forward, mcp_to_pip, palm_normal)
        front = _angle_between(wrist_to_mcp, mcp_to_pip)
        pip_flex = _finger_flex(points, mcp, pip, dip)
        dip_flex = _finger_flex(points, pip, dip, tip)

        angles[f"{prefix}MCP_side_joint"] = _clamp_rad(side, -10.0, 10.0)
        angles[f"{prefix}MCP_front_joint"] = _clamp_rad(front, 0.0, 100.0)
        angles[f"{prefix}PIP_joint"] = _clamp_rad(pip_flex, 0.0, 90.0)
        angles[f"{prefix}DIP_joint"] = _clamp_rad(dip_flex, 0.0, 90.0)

    add_standard_finger("I", 5, 6, 7, 8)
    add_standard_finger("M", 9, 10, 11, 12)
    add_standard_finger("R", 13, 14, 15, 16)

    # Pinkie has one extra metacarpal joint
    pinkie_metacarpal = _angle_between(points[17] - wrist, points[18] - points[17])
    pinkie_side = _signed_angle_on_plane(palm_forward, points[18] - points[17], palm_normal)
    pinkie_front = max(0.0, _angle_between(points[17] - wrist, points[18] - points[17]) - 0.5 * pinkie_metacarpal)
    pinkie_pip = _finger_flex(points, 17, 18, 19)
    pinkie_dip = _finger_flex(points, 18, 19, 20)

    angles["metacarpal_joint"] = _clamp_rad(pinkie_metacarpal, 0.0, 45.0)
    angles["PMCP_side_joint"] = _clamp_rad(pinkie_side, -10.0, 10.0)
    angles["PMCP_front_joint"] = _clamp_rad(pinkie_front, 0.0, 100.0)
    angles["PPIP_joint"] = _clamp_rad(pinkie_pip, 0.0, 90.0)
    angles["PDIP_joint"] = _clamp_rad(pinkie_dip, 0.0, 90.0)

    # Thumb
    thumb_mcp = points[1]
    thumb_pip = points[2]
    thumb_dip = points[3]
    thumb_tip = points[4]

    thumb_base_dir = thumb_pip - thumb_mcp
    thumb_front = _angle_between(thumb_mcp - wrist, thumb_base_dir)
    thumb_rotation = _signed_angle_on_plane(palm_forward, thumb_base_dir, palm_normal)

    thumb_mid_dir = thumb_dip - thumb_pip
    thumb_tip_dir = thumb_tip - thumb_dip
    thumb_side = math.asin(float(np.clip(np.dot(_normalize(thumb_mid_dir), palm_normal), -1.0, 1.0)))
    thumb_mid_flex = _angle_between(thumb_base_dir, thumb_mid_dir)
    thumb_dist_flex = _angle_between(thumb_mid_dir, thumb_tip_dir)

    angles["TMCP_rotation_joint"] = _clamp_rad(thumb_rotation, -60.0, 60.0)
    angles["TMCP_front_joint"] = _clamp_rad(thumb_front, 0.0, 70.0)
    angles["TPIP_side_joint"] = _clamp_rad(thumb_side, -30.0, 30.0)
    angles["TPIP_front_joint"] = _clamp_rad(thumb_mid_flex, -12.0, 12.0)
    angles["TDIP_joint"] = _clamp_rad(thumb_dist_flex, 0.0, 90.0)

    return angles


def get_simple_joint_order() -> List[str]:
    return [
        "IMCP_side_joint", "IMCP_front_joint", "IPIP_joint", "IDIP_joint",
        "MMCP_side_joint", "MMCP_front_joint", "MPIP_joint", "MDIP_joint",
        "RMCP_side_joint", "RMCP_front_joint", "RPIP_joint", "RDIP_joint",
        "metacarpal_joint", "PMCP_side_joint", "PMCP_front_joint", "PPIP_joint", "PDIP_joint",
        "TMCP_rotation_joint", "TMCP_front_joint", "TPIP_side_joint", "TPIP_front_joint", "TDIP_joint",
    ]


def angles_dict_to_array(angles_dict: Dict[str, float]) -> np.ndarray:
    order = get_simple_joint_order()
    return np.array([angles_dict[name] for name in order], dtype=np.float64)
