"""yjh/infer.py

该脚本是 yjh 目录下的主推理入口，用于对一个步态视频（或其抽帧图片序列）完成：

1) 人体检测 + 2D 姿态估计（MMPose + MMDet，body8 关键点）
2) 基于检测框的分割（Segment Anything, SAM）得到人体 mask
3) mask 后处理 + 几何特征提取（大腿/小腿长度与宽度、足尖 toe 估计）
4) 根据关节坐标序列计算关节角度曲线、步态事件（HS/TO）与相位比例
5) 生成角度变化图帧序列，并与预测可视化帧拼接生成结果视频

本文件原始版本把所有逻辑都写在顶层，且存在大量重复 import / 硬编码路径。
为提高可读性，这里做了“以可读性为主、尽量不改输出”的整理：

- 把顶层执行逻辑收敛到 `main()`，避免 import 时自动跑推理
- 用 `InferConfig` 管理关键路径/权重配置（目前仍保留默认值，便于对照原实现）
- 将主要步骤拆为若干小函数，并补充中文注释与必要的类型提示

说明：你当前环境不要求跑通，所以这里的目标是**逻辑更清晰、结构更可维护**。
"""

from __future__ import annotations

# ========================
# 0. 环境与依赖导入
# ========================

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# 可选：限制可见 GPU。若不需要，请注释该行。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# 让脚本在“从仓库根目录/任意位置运行”时，也能 import 到 mmpose-main 内部模块。
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 原作者环境里写死了 /home/yjh/...；这里改成“可选追加”。
# 用法示例（Linux/Mac）：
#   export SEGMENT_ANYTHING_DIR=/path/to/segment-anything-main
#   export MMPOSE_DIR=/path/to/mmpose-main
for _env in ("SEGMENT_ANYTHING_DIR", "MMPOSE_DIR"):
    _p = os.environ.get(_env)
    if _p and _p not in sys.path:
        sys.path.append(_p)

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

print("torch version:", torch.__version__, torch.cuda.is_available())
print("torchvision version:", torchvision.__version__)

import mmpose

print("mmpose version:", mmpose.__version__)
print("cuda version:", get_compiling_cuda_version())
print("compiler information:", get_compiler_version())

# --- 第三方推理 API ---
from segment_anything import SamPredictor, sam_model_registry

# --- 本项目（yjh）后处理模块 ---
from demo.topdown_demo_with_mmdet_new import mmpose as MMPoseDetPoseWrapper
from pic2vid import getvid, sort_by_number
from yjh.angle import find_angle
from yjh.footpoint_new import process_frame_footpoint
from yjh.length_and_width import length_and_width


# ========================
# 1. 配置定义
# ========================


@dataclass(frozen=True)
class InferConfig:
    """推理配置。

    说明：为了最大程度保持与原脚本一致，这里仍保留原作者的默认路径。
    你可以在本机改成相对路径或 Windows 路径。
    """

    # 数据集/输出根目录（原作者硬编码在 /home/yjh/...）
    work_dir: Path = Path("/home/yjh/code_yjh/mmpose-main/yjh")
    videoname: str = "walk_woman_processed_1"

    # SAM 权重
    sam_checkpoint: Path = Path("/home/yjh/code_yjh/sam_vit_h_4b8939.pth")
    sam_model_type: str = "vit_h"
    device: str = "cuda"

    # det + pose 配置（来自 demo/topdown_demo_with_mmdet_new.py 的封装）
    det_config: str = "/home/yjh/code_yjh/mmpose-main/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
    det_checkpoint: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
    pose_config: str = "/home/yjh/code_yjh/mmpose-main/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py"
    pose_checkpoint: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"


def build_paths(cfg: InferConfig) -> Dict[str, Path]:
    """根据 videoname 构建输入/输出路径。"""

    base = cfg.work_dir / "ori" / cfg.videoname
    return {
        "base": base,
        "pic": base / "pic",
        "mask": base / "mask",
        "skeleton": base / "skeleton",
        "pred": base / "pred",
        "pred1": base / "pred1",
        "chart": base / "chart",
        "result_dir": cfg.work_dir / "result",
    }


def ensure_dirs(paths: Dict[str, Path]) -> None:
    """创建必要目录（若已存在则忽略）。"""

    for k in ("pic", "mask", "skeleton", "pred", "pred1", "chart", "result_dir"):
        paths[k].mkdir(parents=True, exist_ok=True)


def ankle_Dorsiflexion(
    left_HS: Sequence[int],
    right_HS: Sequence[int],
    coords: np.ndarray,
) -> Tuple[List[float], List[float]]:
    """踝背屈角（示例实现，按原脚本逻辑保留）。

    说明：该函数不是标准的“踝关节角度定义”，而是按原作者的构造方法：
    - 以 toe 点和一个“水平参考点”构造三点角，计算足部相对水平的某种背屈角。
    - left/right 分别在 HS（Heel Strike）事件帧上计算。
    """

    left_angles: List[float] = []
    right_angles: List[float] = []

    # coords: [T, 8, 2]
    # 2: left ankle, 6: left toe
    for i in left_HS:
        point_a = (coords[i, 2, 0], float(np.max(coords[:, 6, 1])))
        point_b = coords[i, 6, :]
        point_c = (point_a[0] + 2, point_a[1])
        left_angles.append(float(find_angle(point_c, point_b, point_a)))

    # 5: right ankle, 7: right toe
    for t in right_HS:
        point_a = (coords[t, 5, 0], float(np.max(coords[:, 7, 1])))
        point_b = coords[t, 7, :]
        point_c = (point_a[0] + 2, point_a[1])
        right_angles.append(float(find_angle(point_c, point_b, point_a)))

    return left_angles, right_angles
        
        

def getresult(
    hip_point: np.ndarray,
    knee_point: np.ndarray,
    ankle_point: np.ndarray,
    toe_point: Tuple[int, int],
    thigh_length: float,
    thigh_width: float,
    calf_length: float,
    calf_width: float,
) -> List[float]:
    """计算单侧下肢的关键输出（角度 + y坐标 + 长宽）。"""

    perp_point = (hip_point[0], hip_point[1] + 2)
    ankle_angle = find_angle(toe_point, knee_point, ankle_point)
    knee_angle = find_angle(hip_point, ankle_point, knee_point)
    hip_angle = find_angle(knee_point, perp_point, hip_point)

    hip_y = hip_point[1]
    knee_y = knee_point[1]
    ankle_y = ankle_point[1]

    return [
        float(ankle_angle),
        float(knee_angle),
        float(hip_angle),
        float(hip_y),
        float(knee_y),
        float(ankle_y),
        float(thigh_length),
        float(thigh_width),
        float(calf_length),
        float(calf_width),
    ]

def getcycletime(time_cycle):
    if len(time_cycle) <2:
        print('未录制到完整周期，请再次录制')
        return 0
    elif len(time_cycle) <4 and len(time_cycle) > 1:
        return time_cycle[-1]-time_cycle[0]
    else:
        return time_cycle[-2]-time_cycle[1]
    
def stance_and_swing(TO,HS,time_cycle):
    key=time_cycle[-1]
    HS_new = [x for x in HS if x > key]
    if not HS_new:
        key=time_cycle[-2]
        HS_new = [x for x in HS if x > key]
    nearest_TO = min(TO, key=lambda x: abs(x - key))
    nearest_HS = min(HS_new, key=lambda x: x - key)
    return nearest_TO,nearest_HS
    # for i in HS:
    #     if i > TO[-2] and i < TO[-1] :
    #         return i-TO[-2] , TO[-1] -i,i
        
def init_sam_predictor(cfg: InferConfig) -> SamPredictor:
    """初始化 SAM predictor。"""

    sam = sam_model_registry[cfg.sam_model_type](checkpoint=str(cfg.sam_checkpoint))
    sam.to(device=cfg.device)
    return SamPredictor(sam)


def init_det_pose_wrapper(cfg: InferConfig) -> object:
    """初始化 det+pose wrapper（来自 demo/topdown_demo_with_mmdet_new.py）。"""

    return MMPoseDetPoseWrapper(
        args_input={
            "det_config": cfg.det_config,
            "det_checkpoint": cfg.det_checkpoint,
            "pose_config": cfg.pose_config,
            "pose_checkpoint": cfg.pose_checkpoint,
            "draw_heatmap": False,
            "skeleton_style": "mmpose",
        }
    )


def run_inference_on_frames(
    cfg: InferConfig,
    paths: Dict[str, Path],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """对帧序列做逐帧推理与特征提取。

    返回：
        coords: [T, 8, 2]
        results: [T, 2, 10]
        frame_files: 排好序的帧文件名列表（仅文件名，不含目录）
    """

    pic_dir = paths["pic"]
    frame_files = sorted(os.listdir(pic_dir), key=sort_by_number)
    if not frame_files:
        raise RuntimeError(f"pic 目录为空：{pic_dir}")

    coords = np.zeros((len(frame_files), 8, 2), dtype=float)
    results = np.zeros((len(frame_files), 2, 10), dtype=float)

    predictor = init_sam_predictor(cfg)
    detect = init_det_pose_wrapper(cfg)

    # 存储上一帧以及最后一次非重叠帧的状态信息（用于 toe 点在重叠时保持连续）。
    previous_frame_data: Dict[str, object] = {
    'toe_left': None,
    'knee_left': None,
    'ank_left': None,
    'toe_right': None,
    'knee_right': None,
    'ank_right': None,
    
    # 最后一次有效计算出的相对角度 (足膝连线 vs 踝膝连线)
    # The last validly calculated relative angle (toe-knee line vs. ankle-knee line)
    'last_known_relative_angle_left': 0,
    'last_known_relative_angle_right': 0,

    # 相对角度的角速度
    # Angular velocity of the relative angle
    'angular_velocity_left': 0,
    'angular_velocity_right': 0,

    # 最后一次有效计算出的足尖到膝盖的距离
    # The last validly calculated distance from toe to knee
    'toe_knee_radius_left': 0,
    'toe_knee_radius_right': 0
    }

    for frame_idx, filename in enumerate(frame_files, start=1):
        # 文件名通常形如 00001.jpg，这里沿用原作者的 idx-1 写入策略。
        idx = int(Path(filename).stem)
        arr_index = idx - 1

        img_path = pic_dir / filename
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] 无法读取图像：{img_path}")
            continue

        # SAM predictor 期望 RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # det + pose，输出可视化写到 pred 目录
        pred_out_path = paths["pred"] / filename
        pred_instances, bbox = detect.run(str(img_path), str(pred_out_path))

        # 关键点索引（COCO）：髋 11/12，膝 13/14，踝 15/16
        hip_left, hip_right = pred_instances.keypoints[0][11], pred_instances.keypoints[0][12]
        knee_left, knee_right = pred_instances.keypoints[0][13], pred_instances.keypoints[0][14]
        ank_left, ank_right = pred_instances.keypoints[0][15], pred_instances.keypoints[0][16]

        # --- SAM 分割 ---
        predictor.set_image(img_rgb)
        masks, score, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=False,
        )
        mask = masks[0]
        mask_uint8 = mask.astype(np.uint8) * 255

        # 闭运算：填洞/连接断裂
        morph_kernel_size = 5
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        processed_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(str(paths["mask"] / filename), processed_mask)

        # --- 长宽估计 ---
        thigh_length_left, thigh_width_left = length_and_width(img_rgb, processed_mask, hip_left, knee_left)
        thigh_length_right, thigh_width_right = length_and_width(img_rgb, processed_mask, hip_right, knee_right)
        calf_length_left, calf_width_left = length_and_width(img_rgb, processed_mask, knee_left, ank_left)
        calf_length_right, calf_width_right = length_and_width(img_rgb, processed_mask, knee_right, ank_right)

        # --- toe 点估计（含重叠预测）---
        toe_left, toe_right, previous_frame_data = process_frame_footpoint(
            processed_mask,
            ank_left,
            knee_left,
            calf_length_left,
            ank_right,
            knee_right,
            calf_length_right,
            previous_frame_data,
        )

        # --- 可视化：在 pred 图上画 toe 点与 toe-ankle 连线 ---
        pred_img = cv2.imread(str(pred_out_path))
        if pred_img is not None:
            cv2.circle(pred_img, (int(toe_left[0]), int(toe_left[1])), 4, (0, 0, 255), -1)
            cv2.circle(pred_img, (int(toe_right[0]), int(toe_right[1])), 4, (0, 0, 255), -1)
            cv2.line(
                pred_img,
                (int(toe_left[0]), int(toe_left[1])),
                (int(ank_left[0]), int(ank_left[1])),
                (85, 255, 0),
                2,
            )
            cv2.line(
                pred_img,
                (int(toe_right[0]), int(toe_right[1])),
                (int(ank_right[0]), int(ank_right[1])),
                (0, 120, 255),
                2,
            )
            cv2.imwrite(str(paths["pred1"] / filename), pred_img)

        # --- 结果落盘：coords/results ---
        coords[arr_index] = [
            hip_left,
            knee_left,
            ank_left,
            hip_right,
            knee_right,
            ank_right,
            toe_left,
            toe_right,
        ]

        result_left = getresult(
            hip_left,
            knee_left,
            ank_left,
            toe_left,
            thigh_length_left,
            thigh_width_left,
            calf_length_left,
            calf_width_left,
        )
        result_right = getresult(
            hip_right,
            knee_right,
            ank_right,
            toe_right,
            thigh_length_right,
            thigh_width_right,
            calf_length_right,
            calf_width_right,
        )
        results[arr_index] = np.array([result_left, result_right], dtype=float)

        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{len(frame_files)} frames")

    return coords, results, frame_files


def save_npy_results(cfg: InferConfig, paths: Dict[str, Path], coords: np.ndarray, results: np.ndarray) -> None:
    """保存 npy 文件到 result 目录。"""

    np.save(str(paths["result_dir"] / f"{cfg.videoname}_coords.npy"), coords)
    np.save(str(paths["result_dir"] / f"{cfg.videoname}_results.npy"), results)
# np.save('/home/yjh/code_yjh/mmpose-main/yjh/result/walk_woman_processed_2_bboxes.npy',bboxes)
# x_hip_left,y_hip_left=hip_left[0],hip_left[1]
    # x_hip_right,y_hip_right=hip_right[0],hip_right[1]
    # x_knee_left,y_knee_left=knee_left[0],knee_left[1]
    # x_knee_right,y_knee_right=knee_right[0],knee_right[1]
    # x_ank_left,y_ank_left=ank_left[0],ank_left[1]
    # x_ank_right,y_ank_right=ank_right[0],ank_right[1]
    # Thigh_length_left=math.sqrt((x_hip_left-x_knee_left)**2+(y_hip_left-y_knee_left)**2)
    # Thigh_length_right=math.sqrt((x_hip_right-x_knee_right)**2+(y_hip_right-y_knee_right)**2)
    # calf_length_left=math.sqrt((x_knee_left-x_ank_left)**2+(y_knee_left-y_ank_left)**2)
    # calf_length_right=math.sqrt((x_knee_right-x_ank_right)**2+(y_knee_right-y_ank_right)**2)
    # Thigh_middle_left=((x_hip_left+x_knee_left)/2,(y_hip_left+y_knee_left)/2)
    # Thigh_middle_right=((x_hip_right+x_knee_right)/2,(y_hip_right+y_knee_right)/2)
    # calf_middle_left=((x_knee_left+x_ank_left)/2,(y_knee_left+y_ank_left)/2)
    # calf_middle_right=((x_knee_right+x_ank_right)/2,(y_knee_right+y_ank_right)/2)

  

  

    # dect=mmdetection()
    # input_box = pred_instances.bboxes[0]
    # 通过调用`SamPredictor.set_image`来处理图像以产生一个图像嵌入。`SamPredictor`会记住这个嵌入，并将其用于随后的掩码预测。
    # input_box=np.array([  8.077637, 285.1402 ,1155.2648 ,737.1334 ])
    # input_box=np.array([  13.510618 , 4.175497 ,157.63347 , 221.33658 ])
# bboxes=np.load('/home/yjh/code_yjh/mmpose-main/yjh/result/walk_woman_processed_2_bboxes.npy')

# coords=np.load('/home/yjh/code_yjh/mmpose-main/yjh/result/'+videoname+'_coords.npy')

def print_length_width_summary(results: np.ndarray) -> None:
    """打印长度/宽度统计（按原脚本输出保留）。"""

    print('Thigh_length_left:', max(results[:, 0, 6]))
    print('Thigh_width_left:', np.mean(results[:, 0, 7]))
    print('Thigh_length_right:', max(results[:, 1, 6]))
    print('Thigh_width_right:', np.mean(results[:, 1, 7]))
    print('calf_length_left:', max(results[:, 0, 8]))
    print('calf_width_left:', np.mean(results[:, 0, 9]))
    print('calf_length_right:', max(results[:, 1, 8]))
    print('calf_width_right:', np.mean(results[:, 1, 9]))


    # angle_data.add_data(result_left[2])
    # chart.update(angle_data)
    
    # cv2.imwrite(chartpath+i,chart.chart_img)


# np.save('/home/yjh/code_yjh/mmpose-main/yjh/result/walk_woman_processed_2_coords.npy',coords)
# np.save('/home/yjh/code_yjh/mmpose-main/yjh/result/walk_woman_processed_2.npy',results)




def estimate_gait_events_and_cycle(coords: np.ndarray) -> None:
    """根据 coords 估计 HS/TO 与步态周期，并打印相位比例。

    该部分算法沿用原脚本：
    - 对关键点 x 方向相对躯干位移做 Savitzky-Golay 平滑
    - 用 CubicSpline 求导得到速度（原脚本后续并未直接用速度阈值）
    - 用局部极值寻找 TO/HS
    - 再用“膝-髋相对位置符号翻转”作为周期点
    """

    t = np.arange(0, coords.shape[0], 1)
    left_TO: List[int] = []
    left_HS: List[int] = []
    right_TO: List[int] = []
    right_HS: List[int] = []
    time_cycle1: List[int] = []
    time_cycle2: List[int] = []

    # 用髋点均值近似躯干 x（用于消除整体位移）
    v_body = (coords[:, 0, 0] + coords[:, 3, 0]) / 2

    data_smooth_x = np.zeros((6, coords.shape[0]))
    v_x = np.zeros((6, coords.shape[0]))

    # 求出 x 轴速度（仅针对 6 个关节：左右 hip/knee/ank）
    for j in range(6):
        data = coords[:, j, 0] - v_body
        data_smooth = savgol_filter(data, 40, 4)
        cs = CubicSpline(t, data_smooth)
        v = cs.derivative()(t)

        data_smooth_x[j] = data_smooth
        v_x[j] = v

    # 局部极值检测：ankle_x - knee_x 作为脚前后摆动的 proxy
    key_left = data_smooth_x[2, :] - data_smooth_x[1, :]
    key_right = data_smooth_x[5, :] - data_smooth_x[4, :]
    # 步行周期：knee_x - hip_x 由负变正（往前迈步）
    key_cycle_left = data_smooth_x[1, :] - data_smooth_x[0, :]
    key_cycle_right = data_smooth_x[4, :] - data_smooth_x[3, :]

    for m in range(1, v_x.shape[1]):
        # 左脚 TO/HS
        if key_left[m] == np.min(key_left[max(0, m - 5) : min(m + 5, key_left.shape[0])]):
            if m < v_x.shape[1] - 1:
                left_TO.append(m)
        elif key_left[m] == np.max(key_left[max(0, m - 5) : min(m + 5, key_left.shape[0])]):
            if m < v_x.shape[1] - 1:
                left_HS.append(m)

        # 右脚 TO/HS
        if key_right[m] == np.min(key_right[max(0, m - 5) : min(m + 5, key_right.shape[0])]):
            if m < v_x.shape[1] - 1:
                right_TO.append(m)
        elif key_right[m] == np.max(key_right[max(0, m - 5) : min(m + 5, key_right.shape[0])]):
            if m < v_x.shape[1] - 1:
                right_HS.append(m)

        # 周期点（左膝相对左髋由负变正）
        if np.max(key_cycle_left[max(0, m - 5) : m]) < 0 and np.min(key_cycle_left[m : min(m + 5, key_cycle_left.shape[0])]) > 0:
            if m < v_x.shape[1] - 1:
                time_cycle1.append(m)

        # 周期点（右膝相对右髋由负变正）
        if np.max(key_cycle_right[max(0, m - 5) : m]) < 0 and np.min(key_cycle_right[m : min(m + 5, key_cycle_right.shape[0])]) > 0:
            if m < v_x.shape[1] - 1:
                time_cycle2.append(m)

    # 仅打印：踝背屈角序列
    left_angles, right_angles = ankle_Dorsiflexion(left_HS, right_HS, coords)
    print(left_angles, right_angles)

    # 估计周期长度（帧）
    if len(time_cycle1) > 1 and len(time_cycle2) > 1:
        time_cycle = (getcycletime(time_cycle1) + getcycletime(time_cycle2)) / 2
    elif len(time_cycle1) > 1 and len(time_cycle2) <= 1:
        time_cycle = getcycletime(time_cycle1)
    elif len(time_cycle1) <= 1 and len(time_cycle2) > 1:
        time_cycle = getcycletime(time_cycle2)
    else:
        print("未录制到完整周期，请重新录制")
        return

    TO_left_instance, HS_left_instance = stance_and_swing(left_TO, left_HS, time_cycle1)
    TO_right_instance, HS_right_instance = stance_and_swing(right_TO, right_HS, time_cycle2)
    swing_left = (HS_left_instance - TO_left_instance) / time_cycle * 100

    print('完整周期用时:{}帧' .format(time_cycle))
    print('左脚摇摆相占比:{:.3f}%'.format(swing_left))
    print('左脚支撑相占比:{:.3f}%'.format(100-swing_left))
    print('双支撑相占比:{:.3f}%'.format(min(abs(TO_left_instance-HS_right_instance),abs(TO_right_instance-HS_left_instance))/time_cycle*100))
    print('总摇摆相占比:%')

'''#绘结果图
title_list_angle=['ankle_left','ankle_right','knee_left','knee_right','hip_left','hip_right']
title_list_x=['hip_left','knee_left','ank_left','hip_right','knee_right','ank_right']
for idx in range(1,len(results)+1):
    print(idx)
    
    i =  str(idx).zfill(5) + ".jpg"
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    
    # 3行2列
    fig.subplots_adjust(
        left=0.05,      # 左边距
        right=0.95,    # 右边距
        bottom=0.05,    # 底边距
        top=0.95,       # 顶边距
        wspace=0.2,    # 水平子图间距
        hspace=0.3     # 垂直子图间距
    )

    # 为每个子图添加数据并设置坐标轴
    for k, ax in enumerate(axes.flatten()):
        if  k % 2 == 0  :
            data = results[:idx-1,0,k//2]
            data_smooth = savgol_filter(results[:,0,k//2], 40, 4)  
            ax.set_ylim(results[:,0,k//2 ].min()-15,results[:,0,k//2 ].max()+15)
        else:
            data = results[:idx-1,1,k//2 ]
            data_smooth = savgol_filter(results[:,1,k//2], 40, 4)  
            ax.set_ylim(results[:,1,k//2 ].min()-15,results[:,1,k//2 ].max()+15)
        #savgol滤波
        

        
        ax.plot(data,color='blue',label='Raw Signal')
        ax.plot(data_smooth[:idx-1],color='red',label='Smoothed Signal')
        # ax.set_ylim(0, 255)
        ax.set_xlim(0, len(results))
        ax.grid(True)
        ax.legend(loc='upper right',fontsize=6)
        ax.tick_params(labelsize=6)
       
        
        # 设置子图标题（避免重叠关键）
        ax.set_title(title_list_angle[k], fontsize=8,pad=5)  # 减小标题字号
        
        # 仅边缘子图显示标签
        if k not in [0, 3]:
            ax.set_ylabel('')  # 隐藏内部Y轴标签
        if k < 3:
            ax.set_xlabel('')
    plt.savefig('/home/yjh/code_yjh/mmpose-main/yjh/ori/'+videoname+'/chart/'+i)
    plt.clf()
    plt.close()'''
def generate_angle_chart_frames(results: np.ndarray, output_folder: Path) -> None:
    """生成关节角度曲线图帧序列（与原实现一致：3x2 子图）。"""

    title_list_angle = ['ankle_left', 'ankle_right', 'knee_left', 'knee_right', 'hip_left', 'hip_right']
    output_folder.mkdir(parents=True, exist_ok=True)

    # 1) 预计算平滑数据 + y 轴范围（提升生成帧性能）
    print("Pre-calculating data...")
    smoothed_data = np.zeros_like(results)
    y_limits = []
    for k in range(6):
        col = k % 2
        joint_idx = k // 2
        if len(results) >= 40:
            smoothed_data[:, col, joint_idx] = savgol_filter(results[:, col, joint_idx], 40, 4)
        else:
            smoothed_data[:, col, joint_idx] = results[:, col, joint_idx]

        min_val = results[:, col, joint_idx].min()
        max_val = results[:, col, joint_idx].max()
        y_limits.append((min_val - 15, max_val + 15))
    print("Pre-calculation finished.")

    # 2) 创建静态画布
    print("Creating static plot canvas...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.3)
    axes = axes.flatten()

    # 用于存储所有曲线对象的列表（每个子图两条线：raw + smooth）
    lines = []
    for k, ax in enumerate(axes):
        ax.set_ylim(y_limits[k])
        ax.set_xlim(0, len(results))
        ax.grid(True)
        ax.tick_params(labelsize=6)
        ax.set_title(title_list_angle[k], fontsize=8, pad=5)

        # 仅边缘子图显示标签
        if k not in [0, 2, 4]:
            ax.set_ylabel('')
        if k < 4:
            ax.set_xlabel('')

        raw_line, = ax.plot([], [], color='blue', label='Raw Signal')
        smooth_line, = ax.plot([], [], color='red', label='Smoothed Signal')
        lines.append((raw_line, smooth_line))
        ax.legend(loc='upper right', fontsize=6)

    print("Canvas created.")

    # 3) 主循环：逐帧更新曲线并保存
    print("Starting frame generation...")
    total_frames = len(results)
    x_axis_data = np.arange(total_frames)

    for idx in range(1, total_frames + 1):
        if idx % 50 == 0:
            print(f"Processing frame {idx}/{total_frames}")

        for k, (raw_line, smooth_line) in enumerate(lines):
            col = k % 2
            joint_idx = k // 2
            raw_line.set_data(x_axis_data[:idx], results[:idx, col, joint_idx])
            smooth_line.set_data(x_axis_data[:idx], smoothed_data[:idx, col, joint_idx])

        filename = str(idx).zfill(5) + ".jpg"
        plt.savefig(str(output_folder / filename), dpi=100)

    plt.close(fig)
    print("All frames generated.")


def make_result_video(cfg: InferConfig, paths: Dict[str, Path]) -> None:
    """把 pred1 与 chart 拼接为结果视频。"""

    pred1_dir = str(paths["pred1"]) + os.sep
    chart_dir = str(paths["chart"]) + os.sep
    videopath = str(paths["result_dir"] / f"{cfg.videoname}.mp4")
    getvid(pred1_dir, chart_dir, videopath)


def main(cfg: InferConfig) -> None:
    start_time = time.time()

    paths = build_paths(cfg)
    ensure_dirs(paths)

    coords, results, frame_files = run_inference_on_frames(cfg, paths)
    save_npy_results(cfg, paths, coords, results)
    print_length_width_summary(results)

    # 步态事件估计（仅打印，不改变输出文件）
    estimate_gait_events_and_cycle(coords)

    # 曲线图帧生成
    generate_angle_chart_frames(results, paths["chart"])

    # 拼接结果视频
    make_result_video(cfg, paths)

    end_time = time.time()
    print('运行时间：', end_time - start_time)


if __name__ == "__main__":
    # 直接运行时采用默认配置。
    # 若你需要更可复用的形态，可在此加入 argparse，或读取 yaml 配置文件。
    main(InferConfig())

# # 膝和踝坐标
# # 左脚
# for m in range(1,data_smooth_x.shape[1]-1):
#     if data_smooth_x[2,m-1]-data_smooth_x[1,m-1]>=data_smooth_x[2,m]-data_smooth_x[1,m] and data_smooth_x[2,m+1]-data_smooth_x[1,m+1]>=data_smooth_x[2,m]-data_smooth_x[1,m]:
#         left_TO.append(m)
#     elif data_smooth_x[2,m-1]-data_smooth_x[1,m-1]<=data_smooth_x[2,m]-data_smooth_x[1,m] and data_smooth_x[2,m+1]-data_smooth_x[1,m+1]<=data_smooth_x[2,m]-data_smooth_x[1,m]:
#         left_HS.append(m)
 
# # 右脚       
# for m in range(1,data_smooth_x.shape[1]-1):
#     if data_smooth_x[3,m-1]-data_smooth_x[2,m-1]>=data_smooth_x[3,m]-data_smooth_x[2,m] and data_smooth_x[3,m+1]-data_smooth_x[2,m+1]>=data_smooth_x[3,m]-data_smooth_x[2,m]:
#         right_TO.append(m)
#     elif data_smooth_x[3,m-1]-data_smooth_x[2,m-1]<=data_smooth_x[3,m]-data_smooth_x[2,m] and data_smooth_x[3,m+1]-data_smooth_x[2,m+1]<=data_smooth_x[3,m]-data_smooth_x[2,m]:
#         right_HS.append(m)           

# # 根据y轴速度判断
# for j in range(coords.shape[1]):
#     data = coords[:,j,1]
#     data_smooth = savgol_filter(data, 30, 4)
#     # data = coords[:,j,0]
#     # data_smooth = savgol_filter(coords[:,j,0], 40, 4)
#     cs = CubicSpline(t,data_smooth)  
#     velocity_spline = cs.derivative()  # 速度函数
#     v = velocity_spline(t)  # 计算时间点上的速度
#     if j==2 :
        
#         for m in range(0,len(v)-2):
#             if v[m]<=0 and v[m+1]>0:
#                 left_TO.append(m+1)
#             elif v[m]>=0 and v[m+1]<0:
#                 left_HS.append(m+1)
#     if j==5 :
        
#         for m in range(0,len(v)-2):
#             if v[m]<=0 and v[m+1]>0:
#                 right_TO.append(m)
#             elif v[m]>=0 and v[m+1]<0:
#                 right_HS.append(m)   
#     data_raw_x[j]=data
#     data_smooth_x[j]=data_smooth 
#     v_x[j]=v  
      
# # 根据x轴最大值判断
# for j in range(coords.shape[1]):
#     data = coords[:,j,1]
#     data_smooth = savgol_filter(data, 40, 4)
#     # data = coords[:,j,0]
#     # data_smooth = savgol_filter(coords[:,j,0], 40, 4)
#     # cs = CubicSpline(t,data_smooth)  
#     # velocity_spline = cs.derivative()  # 速度函数
#     # v = velocity_spline(t)  # 计算时间点上的速度
    
#     if j==2 :
        
#         for m in range(1,len(data_smooth)-1):
#             if data_smooth[m-1]>=data_smooth[m] and data_smooth[m+1]>=data_smooth[m]:
#                 left_TO.append(m)
#             elif data_smooth[m-1]<=data_smooth[m] and data_smooth[m+1]<= data_smooth[m] :
#                 left_HS.append(m)
#     if j==5 :
        
#         for m in range(1,len(data_smooth)-1):
#             if data_smooth[m-1]>=data_smooth[m] and data_smooth[m+1]>=data_smooth[m]:
#                 right_TO.append(m)
#             elif data_smooth[m-1]<=data_smooth[m] and data_smooth[m+1]<= data_smooth[m] :
#                 right_HS.append(m)   
#     data_raw_x[j]=data
#     data_smooth_x[j]=data_smooth 
    # v_x[j]=v      


# for idx in tqdm(range(1,len(coords)+1)):
#     print(idx)
    
#     i =  str(idx).zfill(5) + ".jpg"
    
#     fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    
#     # 3行2列
#     fig.subplots_adjust(
#         left=0.05,      # 左边距
#         right=0.95,    # 右边距
#         bottom=0.05,    # 底边距
#         top=0.95,       # 顶边距
#         wspace=0.2,    # 水平子图间距
#         hspace=0.3     # 垂直子图间距
#     )

#     # 为每个子图添加数据并设置坐标轴
#     for k, ax in enumerate(axes.flatten()):
       
        
        
                
            
        
#         # ax.set_ylim(data_raw_x[k].min()-15,data_raw_x[k].max()+15)
#         # ax.set_ylim(data_smooth_x[k].min()-15,data_smooth_x[k].max()+15)
#         ax.set_ylim(v_x[k].min()-15,v_x[k].max()+15)

        
        

        
#         # ax.plot(data_raw_x[k,:idx-1],color='blue',label='Raw Signal')
#         # ax.plot(data_smooth[:idx-1],color='red',label='Smoothed Signal')
#         ax.plot(v_x[k,:idx-1],color='red',label='v_x')
#         # ax.set_ylim(0, 255)
#         ax.set_xlim(0, len(coords))
#         ax.grid(True)
#         ax.legend(loc='upper right',fontsize=6)
#         ax.tick_params(labelsize=6)
       
        
#         # 设置子图标题（避免重叠关键）
#         ax.set_title(title_list_y[k], fontsize=8,pad=5)  # 减小标题字号
    # plt.figure(figsize=(4,3),facecolor='white')
    
    # plt.subplot(3, 2, 1)
    # plt.plot(results[:idx-1,0,0],color='blue')  #第二维度0是左，1是右
    # plt.title('ankle_left')
    # plt.xlim(0, 255)
    # plt.subplot(3, 2, 2)
    # plt.plot(results[:idx-1,0,1],color='blue')
    # plt.title('knee_left')
    # plt.subplot(3, 2, 3)
    # plt.plot(results[:idx-1,0,2],color='blue')
    # plt.title('hip_left')
    # plt.subplot(3, 2, 4)
    # plt.plot(results[:idx-1,1,0],color='blue')
    # plt.title('toe_left')
    # plt.subplot(3, 2, 5)
    # plt.plot(results[:idx-1,1,1],color='blue')
    # plt.title('ankle_right')
    # plt.subplot(3, 2, 6)
    # plt.plot(results[:idx-1,1,2],color='blue')
    # plt.title('knee_right')
    
    

   
    # plt.savefig('/home/yjh/code_yjh/mmpose-main/yjh/ori/'+videoname+'/chart/knee_left.png')
# for idx in range(1,len(results)+1):
#     i =  str(idx).zfill(5) + ".jpg"

    
    
    
    
#     # 如果没有视频文件，使用摄像头
#     # video_path = 0

#     # angle_data.add_data(results[idx-1,0,2])
        
    
   
#     plt.plot(results[:,0,2],color='blue')
#     plt.scatter(idx,results[idx-1,0,2],marker = '.',c='red')
#     plt.savefig(chartpath+i)
#     plt.clf()
#     print('hhahah')
    # cv2.imwrite(chartpath+i,chart.chart_img)
    
     
    # cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1.png',image)
    # global_toe = (toe_point[0] + crop_box[0], toe_point[1] + crop_box[1])
    # skeleton = morphology.skeletonize(processed_mask/255)
    # skeleton=(skeleton * 255).astype(np.uint8)
    # cv2.imwrite(skeletonpath+i,skeleton)

    # Thigh_length_left,Thigh_width_left=length_and_width(image,processed_mask,hip_left,knee_left)
    # Thigh_length_right,Thigh_width_right=length_and_width(image,processed_mask,hip_right,knee_right)
    # calf_length_left,calf_width_left=length_and_width(image,processed_mask,knee_left,ank_left)
    # calf_length_right,calf_width_right=length_and_width(image,processed_mask,knee_right,ank_right)
    # image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(maskpath+i,processed_mask)






# cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1_mask_new.png',image)


# contours, hierarchy = cv2.findContours(processed_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# # contour_image=cv2.drawContours(image, contours, -1, (0,0,255), thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
# # # image_final=show_mask(masks[0])
# # contour_image=cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)
# # cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1_contour.png',contour_image)
# cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1_mask.png',processed_mask)

# skeleton = zhang_suen_thinning(processed_mask/255)
# cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1_skeleton.png',skeleton*255)

# skeleton0 = morphology.skeletonize(processed_mask/255)
# skeleton0=(skeleton0 * 255).astype(np.uint8)
# cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/2_skeleton.png',skeleton0)
# # 查找骨架端点
# endpoints, branches = find_endpoints_in_skeleton(skeleton0)
# print(endpoints)
# # if endpoints:
# #         for point in endpoints:
# #             # y,x = point #cv2需要传入的是x,y,但find_endpoints_in_skeleton 函数返回的坐标是 (y, x) 顺序（NumPy数组索引顺序）
# #             cv2.circle( skeleton0, point, 3, (255, 0, 0), 3)
# footpoint_right=(857, 386)
# footpoint_left=(1152, 559)
# cv2.circle( image, footpoint_left, 3, (255, 0, 0), 3)
# cv2.circle( image, footpoint_right, 3, (255, 0, 0), 3)
# cv2.line(image, (round(ank_left[0]), round(ank_left[1])),footpoint_left, (255, 255, 0), 1)
# cv2.line(image, (round(ank_right[0]), round(ank_right[1])),footpoint_right, (255, 255, 0), 1)

    
# cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1_final.png',image)
     
    

# cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1_skeleton.png',skeleton0*255)
# img1=cv2.imread('/home/yjh/code_yjh/mmpose-main/yjh/1_skeleton.png')
# contour_image=cv2.drawContours(img1, contours, -1, (0,0,255), thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
# cv2.imwrite('/home/yjh/code_yjh/mmpose-main/yjh/1_skeleton_new.png',contour_image)

# for contour in contours:
#     if len(contour) < 3:  # 至少需要3个点
#         continue
    
#     # 找到梯度最大点
#     max_grad_point = find_max_gradient_point(contour)
    
# print(max_grad_point)
# point_image=cv2.circle(processed_mask,max_grad_point, 7, (0, 0, 255), -1)
    # # 可视化 (可选)
    # vis = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    # cv2.circle(vis, max_grad_point, 7, (0, 0, 255), -1)  # 红色标记最大梯度点
    # cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)  # 绿色绘制轮廓

# img_path = '/home/yjh/code_yjh/mmpose-main/tests/data/coco/000000196141.jpg'   # 将img_path替换给你自己的路径

# # 使用模型别名创建推理器
# # inferencer = MMPoseInferencer('human')

# # # MMPoseInferencer采用了惰性推断方法，在给定输入时创建一个预测生成器
# # result_generator = inferencer(img_path, show=False,out_dir='/home/yjh/code_yjh/mmpose-main/yjh')
# # inferencer = MMPoseInferencer('human')

# # 使用模型配置名构建推理器
# # inferencer = MMPoseInferencer('td-hm_hrnet-w32_8xb64-210e_coco-256x192')


# # 使用模型配置文件和权重文件的路径或 URL 构建推理器
# inferencer = MMPoseInferencer(
#     pose2d='configs/body_2d_keypoint/topdown_heatmap/coco/' \
#            'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
#     pose2d_weights='https://download.openmmlab.com/mmpose/top_down/' \
#                    'hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
# )
# result_generator = inferencer(img_path, pred_out_dir='predictions')
# result = next(result_generator)


'''python demo/inferencer_demo.py \
/home/yjh/code_yjh/mmpose-main/yjh/final.mp4 \
--pose2d human\
--vis-out-dir vis_results/posetrack18'''

'''

python demo/inferencer_demo.py /home/yjh/code_yjh/mmpose-main/yjh/final.mp4 \
    --pose3d human3d --vis-out-dir vis_results/human3d'''

'''
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
    --input /home/yjh/code_yjh/mmpose-main/yjh/ori/walk_woman_processed_2/pic/00121.jpg \
    --output-root=vis_results  --draw-heatmap'''
    
'''
python body3d_img2pose_demo.py configs/rtmdet_m_640-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth /home/wuyapeng/Downloads/mmpose/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth  --input /home/yjh/code_yjh/mmpose-main/yjh/final.mp4 --output-root /home/yjh/code_yjh/mmpose-main/vis_results
python body3d_img2pose_demo.py configs/rtmdet_m_640-8xb32_coco-person.py https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth /home/wuyapeng/Downloads/mmpose/projects/rtmpose3d/configs/rtmw3d-l_8xb64_cocktail14-384x288.py rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth  --input webcam --output-root output --show #（自带摄像头）
 '''
 
'''
 python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth \
    --input /home/yjh/code_yjh/mmpose-main/yjh/1.png  --draw-heatmap --draw-bbox \
    --output-root vis_results/
    '''
