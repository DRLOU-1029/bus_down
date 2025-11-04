"""
单视频乘客检测、ReID与下车计数脚本 (最终自动化版 v2 - 智能过滤)

功能:
- 对指定的单个视频文件进行乘客检测、追踪和ReID。
- 每个被追踪到的乘客都会被赋予一个在本视频内唯一的、稳定的ID。
- (新) 自动分析视频，通过聚类追踪消失点来定义“下车区域”，无需人工划线。
- 继承了鲁棒的ID分配策略（观察期 + 稳定期）。
- 使用自动定义的多边形“下车区域”来检测乘客下车行为并进行计数。

主要改进 (v2):
1.  **区域定义更智能**: 在自动定义下车区域时，加入了两个关键的过滤器。
    - **过滤短轨迹**: 只考虑持续被追踪一定时间（长度）的个体，忽略因遮挡等原因造成的短期追踪。
    - **过滤错误方向**: 只考虑总体运动方向朝向指定出口（默认为左侧）的个体，忽略走向车厢内部后消失的轨迹。
2.  **完全自动化**: 移除了所有手动选择门槛线的GUI，采用算法自动定义下车区域。
3.  **鲁棒性**: 使用DBSCAN聚类算法，能有效识别主要的下车区域并过滤掉因遮挡等原因造成的车厢内追踪中断（噪声点）。
4.  **适应性**: 能够定义任意形状的下车区域（例如弧形门、多个门形成的组合区域），而不仅仅是一条直线。
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import DBSCAN

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
# 自动定义下车区域函数 (v2 - 带有智能过滤逻辑)
# ==============================================================================

def define_disembarkation_zone(video_path_str: str, yolo_model: YOLO, analysis_frames: int = 500,
                               min_track_length: int = 15, exit_direction_vector: tuple = (-1, 0)):
    """
    通过分析视频中追踪ID的消失点来自动定义下车区域。
    v2版本: 集成了轨迹长度和方向的过滤，以提高准确性。

    Args:
        video_path_str (str): 用于分析的视频文件路径。
        yolo_model (YOLO): 已加载的YOLOv8追踪模型。
        analysis_frames (int): 分析视频的前多少帧来确定区域。-1代表整个视频。
        min_track_length (int): 一条轨迹被认为是有效轨迹所需的最小帧数。
        exit_direction_vector (tuple): 一个表示出口大致方向的向量。
                                       (-1, 0) 表示左侧出口。
                                       (1, 0) 表示右侧出口。
                                       (0, 1) 表示下方出口。

    Returns:
        np.ndarray or None: 返回定义了下车区域的多边形（凸包），如果失败则返回None。
    """
    print(f"\n--- 自动定义下车区域 (智能过滤版) ---")
    print(f"  > 正在分析视频: {Path(video_path_str).name}")

    cap = cv2.VideoCapture(video_path_str)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_to_process = analysis_frames if analysis_frames != -1 and analysis_frames < total_frames else total_frames
    print(f"  > 将分析前 {frames_to_process} 帧...")
    pbar = tqdm(total=frames_to_process, desc="  分析消失点")

    track_histories = {}  # {track_id: {'positions': [(x, y), ...]}}
    previous_frame_track_ids = set()

    for i in range(frames_to_process):
        ret, frame = cap.read()
        if not ret: break

        results = yolo_model.track(frame, persist=True, classes=[0], conf=0.35, verbose=False)
        current_frame_track_ids = set()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            for box_coords, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
                track_id = int(track_id)
                current_frame_track_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box_coords)
                person_bottom_center = ((x1 + x2) // 2, y2)

                if track_id not in track_histories:
                    track_histories[track_id] = {'positions': []}
                track_histories[track_id]['positions'].append(person_bottom_center)
        
        previous_frame_track_ids = current_frame_track_ids
        pbar.update(1)

    pbar.close()

    disappearance_points = []
    filtered_by_length = 0
    filtered_by_direction = 0

    print(f"  > 分析完成。开始过滤 {len(track_histories)} 条总轨迹...")
    
    exit_direction_vector = np.array(exit_direction_vector) / np.linalg.norm(exit_direction_vector)

    for tid, history in track_histories.items():
        # 过滤器1: 轨迹长度
        if len(history['positions']) < min_track_length:
            filtered_by_length += 1
            continue

        start_pos = history['positions'][0]
        end_pos = history['positions'][-1]
        
        # 过滤器2: 运动方向
        move_vec = np.array(end_pos) - np.array(start_pos)
        if np.linalg.norm(move_vec) > 1e-6: # 避免除以零
            move_vec_normalized = move_vec / np.linalg.norm(move_vec)
            dot_product = np.dot(move_vec_normalized, exit_direction_vector)
            # 点积 > 0.2 表示运动方向与出口方向的夹角小于 ~78度，允许一定偏差
            if dot_product <= 0.2:
                filtered_by_direction += 1
                continue
        else: # 如果几乎没有移动，则不是有效的下车轨迹
            filtered_by_direction += 1
            continue

        disappearance_points.append(end_pos)

    print(f"  > 过滤结果:")
    print(f"    - 因轨迹太短被过滤: {filtered_by_length}")
    print(f"    - 因运动方向错误被过滤: {filtered_by_direction}")
    print(f"    - 最终用于聚类的有效消失点: {len(disappearance_points)}")

    if len(disappearance_points) < 10:
        print(f"错误: 经过滤后，剩余的消失点太少 ({len(disappearance_points)}个)，无法可靠地定义区域。")
        cap.release()
        return None

    print(f"  > 正在使用DBSCAN进行聚类...")
    db = DBSCAN(eps=85, min_samples=3).fit(disappearance_points)
    labels = db.labels_

    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) == 0:
        print("错误: DBSCAN未能找到任何有效的簇。可能是eps或min_samples参数不合适。")
        cap.release()
        return None

    largest_cluster_label = unique_labels[counts.argmax()]
    print(f"  > 找到最大簇 (Label: {largest_cluster_label})，包含 {counts.max()} 个点。")

    cluster_points = np.array(disappearance_points)[labels == largest_cluster_label]
    disembarkation_zone = cv2.convexHull(cluster_points)
    print("  > 自动定义下车区域成功！")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if ret:
        cv2.drawContours(frame, [disembarkation_zone], -1, (255, 0, 0), 3)
        cv2.putText(frame, "Auto-defined Disembarkation Zone",
                    (disembarkation_zone.squeeze()[0][0], disembarkation_zone.squeeze()[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("Disembarkation Zone Definition Result", frame)
        print("\n请预览自动定义的下车区域。按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cap.release()

    return disembarkation_zone

# ==============================================================================
# 核心逻辑类 (ReID系统与TrackBuffer)
# ==============================================================================

class TrackBuffer:
    """追踪缓冲区 - 负责管理单个追踪轨迹的观察期和ID稳定期"""
    def __init__(self, track_id, observation_frames=5, stability_frames=8, count_threshold_frames=5):
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
        self.count_threshold_frames = count_threshold_frames
        self.frames_in_downside_area = 0
        self.has_counted = False
        self.first_crossed_to_downside = False

    def add_feature(self, feature, current_position):
        self.frame_count += 1
        if feature is not None: self.features.append(feature)
        self.positions_history.append(current_position)
        if len(self.positions_history) > 15:
            self.positions_history = self.positions_history[-15:]
        if not self.is_ready_for_id and self.frame_count >= self.observation_frames and len(self.features) > 0:
            self.is_ready_for_id = True
            self._compute_mean_feature()
        if self.stability_countdown > 0: self.stability_countdown -= 1

    def assign_id(self, new_id):
        self.assigned_id = new_id
        self.stability_countdown = self.stability_frames

    def is_locked(self):
        return self.stability_countdown > 0

    def _compute_mean_feature(self):
        if not self.features: return
        mean_feat = np.mean(self.features, axis=0)
        norm = np.linalg.norm(mean_feat)
        self.mean_feature = mean_feat / norm if norm > 0 else mean_feat

class SingleVideoReIDSystem:
    """单视频ReID系统 - 负责在一个视频内进行ID的分配和管理"""
    def __init__(self, reid_threshold=0.99, gallery_update_alpha=0.1, max_spatial_distance=150):
        self.reid_threshold = reid_threshold
        self.gallery_update_alpha = gallery_update_alpha
        self.max_spatial_distance = max_spatial_distance
        self.gallery = {}
        self.next_id = 1
        self.track_buffers = {}

    def _find_best_match(self, feature, current_position, exclude_ids=None, is_inside_vehicle=True):
        if feature is None or not self.gallery: return None, 0.0
        best_id, best_sim = None, -1.0
        
        for gid, data in self.gallery.items():
            if exclude_ids and gid in exclude_ids: continue
            if is_inside_vehicle and data.get('disembarked', False): continue

            last_pos = data.get('last_position')
            if last_pos is not None:
                dist = np.linalg.norm(np.array(current_position) - np.array(last_pos))
                if dist > self.max_spatial_distance:
                    continue
            
            sim = np.dot(feature, data['feature'])
            if sim > best_sim:
                best_sim, best_id = sim, gid
                
        return best_id, best_sim

    def _update_gallery(self, gid, feature, position):
        if feature is None or gid not in self.gallery: return
        old_feat = self.gallery[gid]['feature']
        alpha = self.gallery_update_alpha
        new_feat = (1 - alpha) * old_feat + alpha * feature
        norm = np.linalg.norm(new_feat)
        self.gallery[gid]['feature'] = new_feat / norm if norm > 0 else new_feat
        self.gallery[gid]['last_position'] = position
    
    def mark_id_as_disembarked(self, global_id):
        if global_id in self.gallery:
            self.gallery[global_id]['disembarked'] = True
            print(f"  [运动逻辑] ReID:{global_id} 已被标记为 '已下车', 不会再分配给车内人员。")

    def process_track(self, track_id, feature, current_position, current_frame_ids=None, is_inside_vehicle=True):
        if track_id not in self.track_buffers:
            self.track_buffers[track_id] = TrackBuffer(track_id)
        buffer = self.track_buffers[track_id]
        buffer.add_feature(feature, current_position)

        if buffer.is_locked() or buffer.assigned_id is not None:
            if buffer.assigned_id is not None:
                self._update_gallery(buffer.assigned_id, feature, current_position)
            return buffer.assigned_id
        
        if not buffer.is_ready_for_id: return None
        mean_feature = buffer.mean_feature
        if mean_feature is None: return None

        best_id, best_sim = self._find_best_match(mean_feature, current_position, exclude_ids=current_frame_ids, is_inside_vehicle=is_inside_vehicle)

        if best_id is not None and best_sim >= self.reid_threshold:
            buffer.assign_id(best_id)
            self._update_gallery(best_id, mean_feature, current_position)
        else:
            new_id = self.next_id
            self.gallery[new_id] = {'feature': mean_feature, 'disembarked': False, 'last_position': current_position}
            self.next_id += 1
            buffer.assign_id(new_id)
        
        return buffer.assigned_id

class SingleVideoProcessor:
    """单视频处理器"""
    def __init__(self, yolo_model, reid_model_path, reid_num_classes):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing system...\n  - Device: {self.device}")
        self.yolo_model = YOLO(yolo_model)
        self.reid_model = RealReIDModel(model_path=reid_model_path, num_classes=reid_num_classes, device=self.device)
        self.conf_threshold = 0.35
        self.reid_system = SingleVideoReIDSystem(reid_threshold=0.99, gallery_update_alpha=0.1, max_spatial_distance=150)
        print(f"  - ReID System initialized with max_spatial_distance = {self.reid_system.max_spatial_distance} pixels.")
        
        self.passenger_count = 0
        self.disembarkation_zone = None
        self.counted_global_ids = set()

    def _extract_feature(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop_top_y = int(y1 + (y2 - y1) * 0.5)
        if crop_top_y >= y2: crop_top_y = y1
        crop = frame[crop_top_y:y2, x1:x2]
        if crop.shape[0] < 20 or crop.shape[1] < 20: return None
        return self.reid_model.extract_feature(crop)

    def is_in_downside_area(self, point):
        if self.disembarkation_zone is None: return False
        return cv2.pointPolygonTest(self.disembarkation_zone, point, False) >= 0

    def run(self, video_path_str: str, save_output: bool = True):
        video_path = Path(video_path_str)
        if not video_path.exists():
            print(f"Error: 视频文件不存在 -> {video_path}")
            return

        # 步骤 1: 使用智能过滤自动定义下车区域
        self.disembarkation_zone = define_disembarkation_zone(
            video_path_str, self.yolo_model,
            analysis_frames=600,       # 分析前600帧
            min_track_length=20,       # 轨迹至少持续20帧
            exit_direction_vector=(-1, 0) # 出口在左侧
        )
        if self.disembarkation_zone is None:
            print("未能自动定义下车区域，程序退出。")
            return
        
        print(f"\n{'='*60}\n处理视频: {video_path.name}\n{'='*60}")
        
        cap = cv2.VideoCapture(video_path_str)
        fps, w, h, total_frames = (int(cap.get(p)) for p in [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT])
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        writer = None
        if save_output:
            output_dir = Path("single_video_output_reid_counting")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{video_path.stem}_auto_zone_v2.mp4"
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            print(f"  > 视频输出至: {output_path}")

        pbar = tqdm(total=total_frames, desc="  Processing Progress")
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            pbar.update(1)
            frame_idx += 1
            
            results = self.yolo_model.track(frame, persist=True, classes=[0], conf=self.conf_threshold, verbose=False)
            vis_frame = frame.copy()
            boxes = results[0].boxes
            current_frame_reid_ids = set()

            cv2.drawContours(vis_frame, [self.disembarkation_zone], -1, (0, 0, 255), 3)
            
            if boxes.id is not None:
                for box_coords, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
                    track_id = int(track_id)
                    x1, y1, x2, y2 = map(int, box_coords)
                    person_bottom_center = ((x1 + x2) // 2, y2)
                    feature = self._extract_feature(vis_frame, box_coords)
                    
                    is_in_zone = self.is_in_downside_area(person_bottom_center)
                    is_currently_inside_vehicle = not is_in_zone
                    
                    global_person_id = self.reid_system.process_track(track_id, feature, person_bottom_center, current_frame_reid_ids, is_inside_vehicle=is_currently_inside_vehicle)
                    
                    buffer = self.reid_system.track_buffers.get(track_id)
                    label, color = "...", (200, 200, 200)
                    
                    if global_person_id is not None:
                        current_frame_reid_ids.add(global_person_id)
                        color = get_color_for_id(global_person_id)
                        label = f"ID:{global_person_id}"
                        if buffer and buffer.is_locked(): label += " (L)"

                        if not buffer.has_counted:
                            if is_in_zone:
                                if not buffer.first_crossed_to_downside: buffer.first_crossed_to_downside = True
                                buffer.frames_in_downside_area += 1
                            else:
                                if buffer.first_crossed_to_downside: buffer.first_crossed_to_downside = False
                                buffer.frames_in_downside_area = 0

                            if buffer.frames_in_downside_area >= buffer.count_threshold_frames and buffer.first_crossed_to_downside:
                                if global_person_id not in self.counted_global_ids:
                                    self.passenger_count += 1
                                    self.counted_global_ids.add(global_person_id)
                                    buffer.has_counted = True
                                    self.reid_system.mark_id_as_disembarked(global_person_id)
                                    cv2.putText(vis_frame, "Counted!", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                                
                    elif buffer and not buffer.is_ready_for_id:
                        label = f"Obs({buffer.frame_count}/{buffer.observation_frames})"
                        color = (128, 128, 128)
                    
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis_frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                    cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

                    if buffer and global_person_id is not None and len(buffer.positions_history) > 1:
                        track_color = (0, 255, 255) if buffer.has_counted else color
                        for i in range(1, len(buffer.positions_history)):
                            cv2.line(vis_frame, buffer.positions_history[i-1], buffer.positions_history[i], track_color, 2)
                        cv2.circle(vis_frame, person_bottom_center, 5, track_color, -1)

            cv2.putText(vis_frame, f"Disembarkation Count: {self.passenger_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            if writer: writer.write(vis_frame)
        
        cap.release()
        pbar.close()
        if writer: writer.release()
        print(f"\n视频处理完成。")
        print(f"  - 共识别到 {self.reid_system.next_id - 1} 个独立个体 (ReID ID)。")
        print(f"  - 总下车人数: {self.passenger_count}")
        print(f"  - 计数的独立ReID ID数量: {len(self.counted_global_ids)}")

def main():
    print(f"\n{'='*60}\n单视频乘客检测、ReID与下车计数脚本 (最终自动化版 v2)\n{'='*60}")

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
    YOLO_MODEL = r"D:\AI\DeepSORT_YOLOv8_Pytorch\best1.pt"
    REID_MODEL_PATH = r"D:\AI\DeepSORT_YOLOv8_Pytorch\detect-track-reid\final_osnet_ain_x1_0_12ids.pth" # 替换为您的ReID模型路径
    REID_NUM_CLASSES = 12 # 这个数字必须和你训练该ReID模型时使用的训练集ID数量完全一致

    if not Path(YOLO_MODEL).exists():
        print(f"错误: 找不到YOLO模型文件 -> {YOLO_MODEL}。请检查 main 函数配置区的路径。")
        return
    
    processor = SingleVideoProcessor(
        yolo_model=YOLO_MODEL,
        reid_model_path=REID_MODEL_PATH,
        reid_num_classes=REID_NUM_CLASSES
    )
    processor.run(str(video_path_str), save_output=True)
    print(f"\n{'='*60}\n所有处理完成！\n{'='*60}")

if __name__ == "__main__":
    main()