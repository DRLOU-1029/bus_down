"""
单视频乘客检测、ReID与下车计数脚本 (左侧区域消失检测版)

功能:
- 对指定的单个视频文件进行乘客检测、追踪和ReID。
- 每个被追踪到的乘客都会被赋予一个在本视频内唯一的、稳定的ID。
- 继承了鲁棒的ID分配策略（观察期 + 稳定期）。
- 基于左侧区域ID消失检测下车行为并进行计数。

改进:
1. 下车人数实时显示在结果视频左上角。
2. 自动检测左侧区域，无需手动绘制门槛线。
3. 使用GUI文件选择器导入视频文件。
4. 优化下车检测逻辑，减少误报。
5. (运动逻辑优化1): 一个已下车的ID不会再被分配给车厢内的人，防止ID"瞬移"回车内导致漏计数。
6. (运动逻辑优化2): 引入空间邻近约束，防止ID在拥挤人群中"跳跃"到另一个人身上。
7. 基于左侧区域消失的下车检测机制。
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from torchreid_model import RealReIDModel


# ==============================================================================
# 可视化工具函数
# ==============================================================================

def get_color_for_id(global_id):
    """为不同的ReID ID分配不同的颜色"""
    colors = [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
        (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
        (180, 105, 255), (0, 255, 127), (255, 99, 71), (100, 149, 237)
    ]
    return colors[global_id % len(colors)]


# ==============================================================================
# 核心逻辑类 (ReID系统与TrackBuffer)
# ==============================================================================

class TrackBuffer:
    """追踪缓冲区 - 负责管理单个追踪轨迹的观察期和ID稳定期"""

    def __init__(self, track_id, observation_frames=5, stability_frames=8, disappearance_threshold=30):
        self.track_id = track_id
        self.features = []
        self.frame_count = 0
        self.observation_frames = observation_frames
        self.stability_frames = stability_frames
        self.is_ready_for_id = False
        self.mean_feature = None
        self.assigned_id = None
        self.stability_countdown = 0
        self.positions_history = []
        self.disappearance_threshold = disappearance_threshold
        self.frames_missing = 0
        self.has_counted = False
        self.last_seen_frame = 0
        self.last_position = None

    def add_feature(self, feature, current_position):
        self.frame_count += 1
        self.last_seen_frame = self.frame_count
        self.last_position = current_position
        if feature is not None:
            self.features.append(feature)
        self.positions_history.append(current_position)
        if len(self.positions_history) > 15:
            self.positions_history = self.positions_history[-15:]
        if not self.is_ready_for_id and self.frame_count >= self.observation_frames and len(self.features) > 0:
            self.is_ready_for_id = True
            self._compute_mean_feature()
        if self.stability_countdown > 0:
            self.stability_countdown -= 1
        self.frames_missing = 0  # 重置消失计数

    def mark_missing(self):
        """标记该ID在当前帧未检测到"""
        self.frames_missing += 1

    def is_missing_too_long(self):
        """判断是否消失时间过长"""
        return self.frames_missing >= self.disappearance_threshold

    def assign_id(self, new_id):
        self.assigned_id = new_id
        self.stability_countdown = self.stability_frames

    def is_locked(self):
        return self.stability_countdown > 0

    def _compute_mean_feature(self):
        if not self.features:
            return
        mean_feat = np.mean(self.features, axis=0)
        norm = np.linalg.norm(mean_feat)
        self.mean_feature = mean_feat / norm if norm > 0 else mean_feat


class SingleVideoReIDSystem:
    """单视频ReID系统 - 负责在一个视频内进行ID的分配和管理"""

    def __init__(self, reid_threshold=0.99, gallery_update_alpha=0.1, max_spatial_distance=150):
        self.reid_threshold = reid_threshold
        self.gallery_update_alpha = gallery_update_alpha
        self.max_spatial_distance = max_spatial_distance  # 同一个ID在连续帧之间允许的最大像素移动距离
        self.gallery = {}  # {id: {'feature': ..., 'disembarked': False, 'last_position': (x, y), 'last_seen_frame': int}}
        self.next_id = 1
        self.track_buffers = {}

    def _find_best_match(self, feature, current_position, exclude_ids=None, is_inside_vehicle=True):
        """在特征库中寻找最佳匹配，同时满足外观相似性和空间邻近性"""
        if feature is None or not self.gallery:
            return None, 0.0
        best_id, best_sim = None, -1.0

        for gid, data in self.gallery.items():
            if exclude_ids and gid in exclude_ids:
                continue
            if is_inside_vehicle and data.get('disembarked', False):
                continue

            # 核心改动：空间邻近门控
            last_pos = data.get('last_position')
            if last_pos is not None:
                dist = np.linalg.norm(np.array(current_position) - np.array(last_pos))
                if dist > self.max_spatial_distance:
                    continue  # 距离太远，跳过，防止ID跳跃

            sim = np.dot(feature, data['feature'])
            if sim > best_sim:
                best_sim, best_id = sim, gid

        return best_id, best_sim

    def _update_gallery(self, gid, feature, position, frame_idx):
        """更新特征库中的对应ID特征和最后位置"""
        if feature is None or gid not in self.gallery:
            return

        # 更新特征 (移动平均)
        old_feat = self.gallery[gid]['feature']
        alpha = self.gallery_update_alpha
        new_feat = (1 - alpha) * old_feat + alpha * feature
        norm = np.linalg.norm(new_feat)
        self.gallery[gid]['feature'] = new_feat / norm if norm > 0 else new_feat

        # 更新位置和最后出现帧
        self.gallery[gid]['last_position'] = position
        self.gallery[gid]['last_seen_frame'] = frame_idx

    def mark_id_as_disembarked(self, global_id):
        if global_id in self.gallery:
            self.gallery[global_id]['disembarked'] = True
            print(f"  [下车逻辑] ReID:{global_id} 已被标记为 '已下车', 不会再分配给车内人员。")

    def process_track(self, track_id, feature, current_position, current_frame_ids=None, is_inside_vehicle=True,
                      frame_idx=0):
        """处理单个追踪轨迹，分配或更新ReID ID"""
        if track_id not in self.track_buffers:
            self.track_buffers[track_id] = TrackBuffer(track_id)
        buffer = self.track_buffers[track_id]
        buffer.add_feature(feature, current_position)

        if buffer.is_locked() or buffer.assigned_id is not None:
            if buffer.assigned_id is not None:
                self._update_gallery(buffer.assigned_id, feature, current_position, frame_idx)
            return buffer.assigned_id

        if not buffer.is_ready_for_id:
            return None

        mean_feature = buffer.mean_feature
        if mean_feature is None:
            return None

        best_id, best_sim = self._find_best_match(
            mean_feature, current_position,
            exclude_ids=current_frame_ids,
            is_inside_vehicle=is_inside_vehicle
        )

        if best_id is not None and best_sim >= self.reid_threshold:
            buffer.assign_id(best_id)
            self._update_gallery(best_id, mean_feature, current_position, frame_idx)
        else:
            new_id = self.next_id
            self.gallery[new_id] = {
                'feature': mean_feature,
                'disembarked': False,
                'last_position': current_position,
                'last_seen_frame': frame_idx
            }
            self.next_id += 1
            buffer.assign_id(new_id)

        return buffer.assigned_id

    def update_missing_tracks(self, current_frame_idx, active_track_ids):
        """更新所有未在当前帧检测到的轨迹"""
        for track_id, buffer in self.track_buffers.items():
            if track_id not in active_track_ids:
                buffer.mark_missing()


class SingleVideoProcessor:
    """单视频处理器"""

    def __init__(self, yolo_model, reid_model_path, reid_num_classes):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing system...\n  - Device: {self.device}")
        print(f"  - Loading YOLO model: {yolo_model}")
        self.yolo_model = YOLO(yolo_model)
        print(f"  - Loading ReID model: {reid_model_path}")
        self.reid_model = RealReIDModel(
            model_path=reid_model_path,
            num_classes=reid_num_classes,
            device=self.device
        )
        self.conf_threshold = 0.35
        # 初始化ReID系统，并设置关键参数
        self.reid_system = SingleVideoReIDSystem(
            reid_threshold=0.99,
            gallery_update_alpha=0.1,
            max_spatial_distance=150  # 关键参数！根据视频分辨率和人物移动速度调整
        )
        print(
            f"  - ReID System initialized with max_spatial_distance = {self.reid_system.max_spatial_distance} pixels.")

        self.passenger_count = 0
        self.counted_global_ids = set()
        self.left_region_ratio = 0.3  # 左侧区域占整个视频宽度的比例
        self.disappearance_threshold = 30  # ID消失多少帧后判定为下车

    def _extract_feature(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop_top_y = int(y1 + (y2 - y1) * 0.5)
        if crop_top_y >= y2:
            crop_top_y = y1
        crop = frame[crop_top_y:y2, x1:x2]
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            return None
        return self.reid_model.extract_feature(crop)

    def is_in_left_region(self, point, frame_width):
        """判断点是否在左侧区域"""
        left_boundary = frame_width * self.left_region_ratio
        return point[0] < left_boundary

    def check_disembarked_ids(self, current_frame_idx, frame_width):
        """检查是否有ID在左侧区域消失时间过长，判定为下车"""
        disembarked_this_frame = 0

        for global_id, data in self.reid_system.gallery.items():
            if global_id in self.counted_global_ids:
                continue

            if data.get('disembarked', False):
                continue

            last_seen_frame = data.get('last_seen_frame', 0)
            last_position = data.get('last_position')

            # 检查是否在左侧区域消失时间过长
            if (last_position is not None and
                    self.is_in_left_region(last_position, frame_width) and
                    current_frame_idx - last_seen_frame >= self.disappearance_threshold):
                self.passenger_count += 1
                self.counted_global_ids.add(global_id)
                self.reid_system.mark_id_as_disembarked(global_id)
                print(
                    f"Frame {current_frame_idx}: Passenger Disembarked! ReID: {global_id} (disappeared in left region)")
                disembarked_this_frame += 1

        return disembarked_this_frame

    def run(self, video_path_str: str, save_output: bool = True):
        video_path = Path(video_path_str)
        if not video_path.exists():
            print(f"Error: 视频文件不存在 -> {video_path}")
            return
        print(f"\n{'=' * 60}\n处理视频: {video_path.name}\n{'=' * 60}")

        cap = cv2.VideoCapture(video_path_str)
        fps, w, h, total_frames = (int(cap.get(p)) for p in
                                   [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                                    cv2.CAP_PROP_FRAME_COUNT])

        # 计算左侧区域边界
        left_boundary = int(w * self.left_region_ratio)
        print(f"  - 左侧下车区域: 0-{left_boundary}像素 (占宽度{self.left_region_ratio * 100}%)")
        print(f"  - 消失判定阈值: {self.disappearance_threshold}帧")

        writer = None
        if save_output:
            output_dir = Path("single_video_output_reid_counting")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{video_path.stem}_left_region_disappearance.mp4"
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f"  > 视频输出至: {output_path}")

        pbar = tqdm(total=total_frames, desc="  Processing Progress")
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            pbar.update(1)
            frame_idx += 1

            results = self.yolo_model.track(frame, persist=True, classes=[0], conf=self.conf_threshold, verbose=False)
            vis_frame = frame.copy()

            # 绘制左侧区域边界线
            cv2.line(vis_frame, (left_boundary, 0), (left_boundary, h), (0, 255, 0), 2)
            cv2.putText(vis_frame, "Left Region", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            boxes = results[0].boxes
            current_frame_reid_ids = set()
            active_track_ids = set()

            if boxes.id is not None:
                for box_coords, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
                    track_id = int(track_id)
                    active_track_ids.add(track_id)

                    x1, y1, x2, y2 = map(int, box_coords)
                    person_bottom_center = ((x1 + x2) // 2, y2)
                    feature = self._extract_feature(vis_frame, box_coords)

                    # 判断是否在左侧区域
                    in_left_region = self.is_in_left_region(person_bottom_center, w)

                    global_person_id = self.reid_system.process_track(
                        track_id, feature, person_bottom_center,
                        current_frame_reid_ids,
                        is_inside_vehicle=not in_left_region,  # 在左侧区域视为不在车内
                        frame_idx=frame_idx
                    )

                    buffer = self.reid_system.track_buffers.get(track_id)
                    label, color = "...", (200, 200, 200)

                    if global_person_id is not None:
                        current_frame_reid_ids.add(global_person_id)
                        color = get_color_for_id(global_person_id)
                        label = f"ID:{global_person_id}"
                        if buffer and buffer.is_locked():
                            label += " (L)"

                        # 如果在左侧区域，用特殊颜色标记
                        if in_left_region:
                            color = (0, 165, 255)  # 橙色表示在左侧区域

                    elif buffer and not buffer.is_ready_for_id:
                        label = f"Obs({buffer.frame_count}/{buffer.observation_frames})"
                        color = (128, 128, 128)

                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis_frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                    cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    if buffer and global_person_id is not None and len(buffer.positions_history) > 1:
                        track_color = (0, 255, 255) if global_person_id in self.counted_global_ids else color
                        for i in range(1, len(buffer.positions_history)):
                            cv2.line(vis_frame, buffer.positions_history[i - 1], buffer.positions_history[i],
                                     track_color, 2)
                        cv2.circle(vis_frame, person_bottom_center, 5, track_color, -1)

            # 更新未检测到的轨迹
            self.reid_system.update_missing_tracks(current_frame_idx=frame_idx, active_track_ids=active_track_ids)

            # 检查下车的ID
            disembarked_count = self.check_disembarked_ids(frame_idx, w)
            if disembarked_count > 0:
                cv2.putText(vis_frame, f"+{disembarked_count} Disembarked!", (w // 2 - 100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.putText(vis_frame, f"Disembarkation Count: {self.passenger_count}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # 显示当前帧信息
            cv2.putText(vis_frame, f"Frame: {frame_idx}", (w - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if writer:
                writer.write(vis_frame)

        cap.release()
        pbar.close()
        if writer:
            writer.release()
        print(f"\n视频处理完成。")
        print(f"  - 共识别到 {self.reid_system.next_id - 1} 个独立个体 (ReID ID)。")
        print(f"  - 总下车人数: {self.passenger_count}")
        print(f"  - 计数的独立ReID ID数量: {len(self.counted_global_ids)}")


def main():
    print(f"\n{'=' * 60}\n单视频乘客检测、ReID与下车计数脚本 (左侧区域消失检测版)\n{'=' * 60}")

    root = tk.Tk()
    root.withdraw()
    video_path_str = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
    )
    if not video_path_str:
        print("未选择视频文件，程序退出。")
        return

    # --- 模型配置 (请确保路径和参数正确) ---
    YOLO_MODEL = r"D:\\BUS project\\DeepSORT_YOLOv8_Pytorch\\best1.pt"  # 替换为您的YOLOv8模型路径
    REID_MODEL_PATH = r"D:\\BUS project\\DeepSORT_YOLOv8_Pytorch\\detect-track-reid\\final_osnet_ain_x1_0_12ids.pth"  # 替换为您的ReID模型路径
    REID_NUM_CLASSES = 12  # 这个数字必须和你训练该ReID模型时使用的训练集ID数量完全一致

    for p in [YOLO_MODEL, REID_MODEL_PATH]:
        if not Path(p).exists():
            print(f"错误: 找不到模型文件 -> {p}。请检查 main 函数配置区的路径。")
            return

    processor = SingleVideoProcessor(
        yolo_model=YOLO_MODEL,
        reid_model_path=REID_MODEL_PATH,
        reid_num_classes=REID_NUM_CLASSES
    )
    processor.run(str(video_path_str), save_output=True)
    print(f"\n{'=' * 60}\n所有处理完成！\n{'=' * 60}")


if __name__ == "__main__":
    main()