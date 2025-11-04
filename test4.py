"""
单视频乘客检测、ReID与下车计数脚本 (最终自动化版 v2.2 - 预热过程可视化)

功能:
- 对指定的单个视频文件进行乘客检测、追踪和ReID。
- 自动分析视频，通过聚类追踪消失点来定义“下车区域”。
- 在聚类前，会生成一张包含所有消失点的调试图，用不同颜色区分有效/无效点。
- 继承了鲁棒的ID分配策略和运动逻辑优化。

主要改进 (v2.2):
1.  **预热过程可视化**: 在分析视频定义下车区域时，会实时弹出一个窗口，展示追踪器的工作过程，
    包括目标的边界框、追踪ID和实时累积的运动轨迹。
2.  **调试可视化**: 保留了v2.1的功能，在预热结束后，会显示一张静态的消失点分析图（绿点/红点）。
3.  **区域定义更智能**: 继承v2的智能过滤器 (过滤短轨迹和错误方向的轨迹)。
4.  **完全自动化**: 无需人工划线或选择。
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
    colors = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (180, 105, 255), (0, 255, 127), (255, 99, 71), (100, 149, 237)]
    return colors[global_id % len(colors)]

# ==============================================================================
# 自动定义下车区域函数 (v2.2 - 带有预热过程可视化)
# ==============================================================================
def define_disembarkation_zone(video_path_str: str, yolo_model: YOLO, analysis_frames: int = 500,
                               min_track_length: int = 15, exit_direction_vector: tuple = (-1, 0),
                               visualize_preheat: bool = True):
    """
    通过分析视频中追踪ID的消失点来自动定义下车区域。
    v2.2: 新增了预热过程的实时可视化。
    """
    print(f"\n--- 自动定义下车区域 (v2.2 - 带预热可视化) ---")
    print(f"  > 正在分析视频: {Path(video_path_str).name}")

    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened(): print("错误: 无法打开视频文件。"); return None
        
    ret, first_frame_for_vis = cap.read()
    if not ret: print("错误: 无法读取视频的第一帧。"); cap.release(); return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = analysis_frames if analysis_frames != -1 and analysis_frames < total_frames else total_frames
    
    print(f"  > 将分析前 {frames_to_process} 帧...")
    pbar = tqdm(total=frames_to_process, desc="  分析消失点")
    track_histories = {}

    if visualize_preheat:
        cv2.namedWindow("Preheat Analysis Visualization", cv2.WINDOW_NORMAL)

    for i in range(frames_to_process):
        ret, frame = cap.read()
        if not ret: break
        
        results = yolo_model.track(frame, persist=True, classes=[0], conf=0.35, verbose=False)
        vis_frame = frame.copy() if visualize_preheat else None

        if results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.id.cpu().numpy()):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2) // 2, y2)
                
                if track_id not in track_histories: track_histories[track_id] = {'positions': []}
                track_histories[track_id]['positions'].append(center)

                if visualize_preheat:
                    # 绘制边界框和ID
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # 青色
                    cv2.putText(vis_frame, f"T:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    # 绘制轨迹
                    if len(track_histories[track_id]['positions']) > 1:
                        pts = np.array(track_histories[track_id]['positions'], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(vis_frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
        
        if visualize_preheat:
            cv2.putText(vis_frame, f"Preheating Phase... Frame: {i+1}/{frames_to_process}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Preheat Analysis Visualization", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户提前跳过预热可视化。")
                visualize_preheat = False # 停止后续帧的可视化以加速
                cv2.destroyAllWindows()
        
        pbar.update(1)
    
    pbar.close()
    if visualize_preheat: cv2.destroyAllWindows()

    valid_points, invalid_points = [], []
    filtered_len, filtered_dir = 0, 0
    exit_vec_norm = np.array(exit_direction_vector) / np.linalg.norm(exit_direction_vector)
    
    print(f"  > 分析完成。开始过滤 {len(track_histories)} 条总轨迹...")
    # (过滤逻辑和之前的版本完全一样)
    for tid, history in track_histories.items():
        end_pos = history['positions'][-1]
        is_valid = True
        if len(history['positions']) < min_track_length:
            filtered_len += 1; is_valid = False
        else:
            move_vec = np.array(end_pos) - np.array(history['positions'][0])
            if np.linalg.norm(move_vec) > 1e-6:
                if np.dot(move_vec / np.linalg.norm(move_vec), exit_vec_norm) <= 0.2:
                    filtered_dir += 1; is_valid = False
            else: filtered_dir += 1; is_valid = False
        if is_valid: valid_points.append(end_pos)
        else: invalid_points.append(end_pos)

    print(f"  > 过滤结果: (因轨迹太短: {filtered_len}) (因运动方向错误: {filtered_dir})")
    print(f"  > 有效消失点: {len(valid_points)} | 无效消失点: {len(invalid_points)}")

    # (调试可视化步骤和之前版本完全一样)
    print("\n--- 正在生成消失点可视化图 (方便调试) ---")
    vis_debug_frame = first_frame_for_vis.copy()
    for point in invalid_points: cv2.circle(vis_debug_frame, point, 4, (0, 0, 255), -1)
    for point in valid_points: cv2.circle(vis_debug_frame, point, 5, (0, 255, 0), -1)
    cv2.putText(vis_debug_frame, "Green: Valid Points (Used for Clustering)", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis_debug_frame, "Red: Invalid Points (Filtered Out)", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Disappearance Points Analysis (Press any key to continue)", vis_debug_frame)
    print("请检查有效(绿色)和无效(红色)的消失点。按任意键继续进行聚类...")
    cv2.waitKey(0); cv2.destroyAllWindows()

    if len(valid_points) < 10:
        print("错误: 经过滤后，有效消失点太少，无法定义区域。"); cap.release(); return None

    # (聚类和生成区域的逻辑和之前版本完全一样)
    print(f"  > 使用 {len(valid_points)} 个有效点进行DBSCAN聚类...")
    db = DBSCAN(eps=85, min_samples=5).fit(valid_points)
    labels = db.labels_
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(counts) == 0: print("错误: DBSCAN未能找到任何有效的簇。"); cap.release(); return None
    largest_cluster_label = unique[counts.argmax()]
    print(f"  > 找到最大簇，包含 {counts.max()} 个点。")
    cluster_points = np.array(valid_points)[labels == largest_cluster_label]
    disembarkation_zone = cv2.convexHull(cluster_points)
    print("  > 自动定义下车区域成功！")
    cv2.drawContours(first_frame_for_vis, [disembarkation_zone], -1, (255, 0, 0), 3)
    cv2.imshow("Final Disembarkation Zone Result", first_frame_for_vis)
    print("\n请预览最终生成的下车区域。按任意键开始正式处理视频...")
    cv2.waitKey(0); cv2.destroyAllWindows()
    
    cap.release()
    return disembarkation_zone

# ==============================================================================
# 核心逻辑类 (这部分代码与之前版本相同，无需改动)
# ==============================================================================
class TrackBuffer:
    def __init__(self, track_id, observation_frames=5, stability_frames=8, count_threshold_frames=5):
        self.track_id, self.observation_frames, self.stability_frames, self.count_threshold_frames = track_id, observation_frames, stability_frames, count_threshold_frames
        self.features, self.frame_count, self.is_ready_for_id, self.mean_feature, self.assigned_id, self.stability_countdown = [], 0, False, None, None, 0
        self.positions_history, self.frames_in_downside_area, self.has_counted, self.first_crossed_to_downside = [], 0, False, False
    def add_feature(self, feature, current_position):
        self.frame_count += 1
        if feature is not None: self.features.append(feature)
        self.positions_history.append(current_position)
        if len(self.positions_history) > 15: self.positions_history.pop(0)
        if not self.is_ready_for_id and self.frame_count >= self.observation_frames and self.features:
            self.is_ready_for_id = True; self._compute_mean_feature()
        if self.stability_countdown > 0: self.stability_countdown -= 1
    def assign_id(self, new_id): self.assigned_id, self.stability_countdown = new_id, self.stability_frames
    def is_locked(self): return self.stability_countdown > 0
    def _compute_mean_feature(self):
        if not self.features: return
        mean_feat = np.mean(self.features, axis=0)
        norm = np.linalg.norm(mean_feat)
        self.mean_feature = mean_feat / norm if norm > 0 else mean_feat

class SingleVideoReIDSystem:
    def __init__(self, reid_threshold=0.99, gallery_update_alpha=0.1, max_spatial_distance=150):
        self.reid_threshold, self.gallery_update_alpha, self.max_spatial_distance = reid_threshold, gallery_update_alpha, max_spatial_distance
        self.gallery, self.next_id, self.track_buffers = {}, 1, {}
    def _find_best_match(self, feature, current_position, exclude_ids=None, is_inside_vehicle=True):
        if feature is None or not self.gallery: return None, 0.0
        best_id, best_sim = None, -1.0
        for gid, data in self.gallery.items():
            if (exclude_ids and gid in exclude_ids) or (is_inside_vehicle and data.get('disembarked', False)): continue
            last_pos = data.get('last_position')
            if last_pos and np.linalg.norm(np.array(current_position) - np.array(last_pos)) > self.max_spatial_distance: continue
            sim = np.dot(feature, data['feature'])
            if sim > best_sim: best_sim, best_id = sim, gid
        return best_id, best_sim
    def _update_gallery(self, gid, feature, position):
        if feature is None or gid not in self.gallery: return
        old_feat = self.gallery[gid]['feature']
        new_feat = (1 - self.gallery_update_alpha) * old_feat + self.gallery_update_alpha * feature
        norm = np.linalg.norm(new_feat)
        self.gallery[gid]['feature'] = new_feat / norm if norm > 0 else new_feat
        self.gallery[gid]['last_position'] = position
    def mark_id_as_disembarked(self, global_id):
        if global_id in self.gallery: self.gallery[global_id]['disembarked'] = True; print(f"  [运动逻辑] ReID:{global_id} 已被标记为 '已下车'")
    def process_track(self, track_id, feature, current_position, current_frame_ids=None, is_inside_vehicle=True):
        if track_id not in self.track_buffers: self.track_buffers[track_id] = TrackBuffer(track_id)
        buffer = self.track_buffers[track_id]
        buffer.add_feature(feature, current_position)
        if buffer.is_locked() or buffer.assigned_id is not None:
            if buffer.assigned_id is not None: self._update_gallery(buffer.assigned_id, feature, current_position)
            return buffer.assigned_id
        if not buffer.is_ready_for_id or buffer.mean_feature is None: return None
        best_id, best_sim = self._find_best_match(buffer.mean_feature, current_position, exclude_ids=current_frame_ids, is_inside_vehicle=is_inside_vehicle)
        if best_id is not None and best_sim >= self.reid_threshold:
            buffer.assign_id(best_id); self._update_gallery(best_id, buffer.mean_feature, current_position)
        else:
            new_id = self.next_id; self.gallery[new_id] = {'feature': buffer.mean_feature, 'disembarked': False, 'last_position': current_position}
            self.next_id += 1; buffer.assign_id(new_id)
        return buffer.assigned_id

class SingleVideoProcessor:
    def __init__(self, yolo_model, reid_model_path, reid_num_classes):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing system...\n  - Device: {self.device}")
        self.yolo_model = YOLO(yolo_model)
        self.reid_model = RealReIDModel(model_path=reid_model_path, num_classes=reid_num_classes, device=self.device)
        self.conf_threshold = 0.35
        self.reid_system = SingleVideoReIDSystem()
        self.passenger_count, self.disembarkation_zone, self.counted_global_ids = 0, None, set()
    def _extract_feature(self, frame, box):
        x1, y1, x2, y2 = map(int, box); h, w = frame.shape[:2]; x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop_top_y = int(y1 + (y2 - y1) * 0.5); crop_top_y = y1 if crop_top_y >= y2 else crop_top_y
        crop = frame[crop_top_y:y2, x1:x2]
        return self.reid_model.extract_feature(crop) if crop.shape[0] > 20 and crop.shape[1] > 20 else None
    def is_in_downside_area(self, point): return cv2.pointPolygonTest(self.disembarkation_zone, point, False) >= 0 if self.disembarkation_zone is not None else False
    def run(self, video_path_str: str, save_output: bool = True):
        video_path = Path(video_path_str)
        if not video_path.exists(): print(f"Error: 视频文件不存在 -> {video_path}"); return
        
        # 在这里调用带有新功能的函数
        self.disembarkation_zone = define_disembarkation_zone(
            video_path_str, self.yolo_model,
            analysis_frames=600,       # 分析前600帧
            min_track_length=20,       # 轨迹至少持续20帧
            exit_direction_vector=(-1, 0), # 出口在左侧
            visualize_preheat=True     # !!激活预热可视化!!
        )
        if self.disembarkation_zone is None: print("未能自动定义下车区域，程序退出。"); return
        
        print(f"\n{'='*60}\n开始处理视频: {video_path.name}\n{'='*60}")
        cap = cv2.VideoCapture(video_path_str)
        fps, w, h, total_frames = (int(cap.get(p)) for p in [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT])
        writer = None
        if save_output:
            output_dir = Path("single_video_output_reid_counting"); output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{video_path.stem}_auto_zone_v2_2.mp4"
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)); print(f"  > 视频输出至: {output_path}")

        pbar = tqdm(total=total_frames, desc="  Processing Progress")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            pbar.update(1)
            results = self.yolo_model.track(frame, persist=True, classes=[0], conf=self.conf_threshold, verbose=False)
            vis_frame = frame.copy()
            cv2.drawContours(vis_frame, [self.disembarkation_zone], -1, (0, 0, 255), 3)
            
            if results[0].boxes.id is not None:
                current_frame_reid_ids = set()
                for box, track_id in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.id.cpu().numpy()):
                    track_id = int(track_id); x1, y1, x2, y2 = map(int, box)
                    center = ((x1 + x2) // 2, y2)
                    feature = self._extract_feature(vis_frame, box)
                    global_id = self.reid_system.process_track(track_id, feature, center, current_frame_reid_ids, is_inside_vehicle=not self.is_in_downside_area(center))
                    buffer = self.reid_system.track_buffers.get(track_id)
                    label, color = "...", (200, 200, 200)
                    
                    if global_id is not None:
                        current_frame_reid_ids.add(global_id)
                        color = get_color_for_id(global_id); label = f"ID:{global_id}"
                        if buffer and buffer.is_locked(): label += " (L)"
                        if not buffer.has_counted:
                            if self.is_in_downside_area(center):
                                if not buffer.first_crossed_to_downside: buffer.first_crossed_to_downside = True
                                buffer.frames_in_downside_area += 1
                            else:
                                if buffer.first_crossed_to_downside: buffer.first_crossed_to_downside = False
                                buffer.frames_in_downside_area = 0
                            if buffer.frames_in_downside_area >= buffer.count_threshold_frames and buffer.first_crossed_to_downside:
                                if global_id not in self.counted_global_ids:
                                    self.passenger_count += 1; self.counted_global_ids.add(global_id); buffer.has_counted = True
                                    self.reid_system.mark_id_as_disembarked(global_id); cv2.putText(vis_frame, "Counted!", (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    elif buffer and not buffer.is_ready_for_id:
                        label = f"Obs({buffer.frame_count}/{buffer.observation_frames})"; color = (128, 128, 128)
                    
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis_frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                    cv2.putText(vis_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(vis_frame, f"Disembarkation Count: {self.passenger_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            if writer: writer.write(vis_frame)
        
        cap.release(); pbar.close();
        if writer: writer.release()
        print(f"\n视频处理完成。\n  - 总下车人数: {self.passenger_count}")

def main():
    print(f"\n{'='*60}\n单视频乘客检测、ReID与下车计数脚本 (v2.2 - 带预热可视化)\n{'='*60}")
    root = tk.Tk(); root.withdraw()
    video_path_str = filedialog.askopenfilename(title="选择视频文件", filetypes=[("视频文件", "*.mp4 *.avi"), ("所有文件", "*.*")])
    if not video_path_str: print("未选择视频文件，程序退出。"); return

    YOLO_MODEL = r"D:\AI\DeepSORT_YOLOv8_Pytorch\best1.pt"
    REID_MODEL_PATH = r"D:\AI\DeepSORT_YOLOv8_Pytorch\detect-track-reid\final_osnet_ain_x1_0_12ids.pth" # 替换为您的ReID模型路径
    REID_NUM_CLASSES = 12
    if not Path(YOLO_MODEL).exists(): print(f"错误: 找不到YOLO模型文件 -> {YOLO_MODEL}"); return
    
    processor = SingleVideoProcessor(yolo_model=YOLO_MODEL, reid_model_path=REID_MODEL_PATH, reid_num_classes=REID_NUM_CLASSES)
    processor.run(str(video_path_str), save_output=True)
    print(f"\n{'='*60}\n所有处理完成！\n{'='*60}")

if __name__ == "__main__":
    main()