"""
单视频乘客检测、ReID与下车计数脚本 (最终优化版)

功能:
- 对指定的单个视频文件进行乘客检测、追踪和ReID。
- 每个被追踪到的乘客都会被赋予一个在本视频内唯一的、稳定的ID。
- 继承了鲁棒的ID分配策略（观察期 + 稳定期）。
- 引入了“门槛线”概念，用于检测乘客下车行为并进行计数。

改进:
1. 下车人数实时显示在结果视频左上角。
2. 支持可视化选择门槛线 (在第一帧上点击两点)。
3. 使用GUI文件选择器导入视频文件。
4. 优化下车检测逻辑，减少误报。
5. 用户手动指定下车侧。
6. (运动逻辑优化1): 一个已下车的ID不会再被分配给车厢内的人，防止ID“瞬移”回车内导致漏计数。
7. (运动逻辑优化2): 引入空间邻近约束，防止ID在拥挤人群中“跳跃”到另一个人身上，解决了ID在下车瞬间被抢走的问题。
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, simpledialog

# 确保 torchreid_model.py 文件在同一个目录下，或者可以被Python找到
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
# 门槛线选择GUI函数
# ==============================================================================

line_points = []
downside_point = None
drawing_line = False
drawing_downside = False
frame_for_selection = None

def select_line_and_side_callback(event, x, y, flags, param):
    global line_points, downside_point, drawing_line, drawing_downside, frame_for_selection

    if drawing_line:
        if event == cv2.EVENT_LBUTTONDOWN:
            line_points.append((x, y))
            if len(line_points) > 2:
                line_points = line_points[-2:]
        elif event == cv2.EVENT_MOUSEMOVE:
            temp_frame = frame_for_selection.copy()
            if len(line_points) == 1:
                cv2.line(temp_frame, line_points[0], (x, y), (0, 255, 0), 2)
            elif len(line_points) == 2:
                cv2.line(temp_frame, line_points[0], line_points[1], (0, 255, 0), 2)
            cv2.putText(temp_frame, "Click 2 points for line, then 'q' to confirm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Select Threshold Line (Press 'q' to confirm)", temp_frame)
        elif event == cv2.EVENT_LBUTTONUP:
            if len(line_points) == 2:
                temp_frame = frame_for_selection.copy()
                cv2.line(temp_frame, line_points[0], line_points[1], (0, 255, 0), 2)
                cv2.putText(temp_frame, "Click 2 points for line, then 'q' to confirm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Select Threshold Line (Press 'q' to confirm)", temp_frame)
    elif drawing_downside:
        if event == cv2.EVENT_LBUTTONDOWN:
            downside_point = (x, y)
            temp_frame = frame_for_selection.copy()
            cv2.line(temp_frame, line_points[0], line_points[1], (0, 255, 0), 2)
            cv2.circle(temp_frame, downside_point, 10, (0, 0, 255), -1)
            cv2.putText(temp_frame, "Click on the 'Disembarkation Side', then 'q' to confirm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Select Disembarkation Side (Press 'q' to confirm)", temp_frame)

def visualize_select_line_and_side(frame):
    global line_points, downside_point, drawing_line, drawing_downside, frame_for_selection
    line_points, downside_point = [], None
    frame_for_selection = frame.copy()

    drawing_line, drawing_downside = True, False
    cv2.imshow("Select Threshold Line (Press 'q' to confirm)", frame_for_selection)
    cv2.setMouseCallback("Select Threshold Line (Press 'q' to confirm)", select_line_and_side_callback)
    print("--- 阶段1: 选择门槛线 ---")
    print("请在图像上点击两点来定义门槛线。按 'r' 重置, 按 'q' 确认。")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and len(line_points) == 2: break
        elif key == ord('r'):
            line_points = []
            frame_for_selection = frame.copy()
            cv2.imshow("Select Threshold Line (Press 'q' to confirm)", frame_for_selection)
            print("点已重置。请选择两个新点。")
        elif key == ord('q') and len(line_points) < 2:
            print("错误: 在按 'q' 确认之前，请准确选择两个点。")
    if len(line_points) < 2:
        print("门槛线选择被取消或无效。")
        cv2.destroyAllWindows()
        return None, None, None

    drawing_line, drawing_downside = False, True
    temp_frame = frame_for_selection.copy()
    cv2.line(temp_frame, line_points[0], line_points[1], (0, 255, 0), 2)
    cv2.imshow("Select Disembarkation Side (Press 'q' to confirm)", temp_frame)
    cv2.setMouseCallback("Select Disembarkation Side (Press 'q' to confirm)", select_line_and_side_callback)
    print("\n--- 阶段2: 选择下车侧 ---")
    print("请在门槛线外侧点击一个点，指示'下车'方向。按 'q' 确认。")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and downside_point is not None: break
        elif key == ord('q') and downside_point is None:
            print("错误: 请点击一个点来指示下车侧。")
    cv2.destroyAllWindows()
    return (line_points[0], line_points[1], downside_point) if line_points and downside_point else (None, None, None)

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
        self.max_spatial_distance = max_spatial_distance  # 同一个ID在连续帧之间允许的最大像素移动距离
        self.gallery = {}  # {id: {'feature': ..., 'disembarked': False, 'last_position': (x, y)}}
        self.next_id = 1
        self.track_buffers = {}

    def _find_best_match(self, feature, current_position, exclude_ids=None, is_inside_vehicle=True):
        """在特征库中寻找最佳匹配，同时满足外观相似性和空间邻近性"""
        if feature is None or not self.gallery: return None, 0.0
        best_id, best_sim = None, -1.0
        
        for gid, data in self.gallery.items():
            if exclude_ids and gid in exclude_ids: continue
            if is_inside_vehicle and data.get('disembarked', False): continue

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

    def _update_gallery(self, gid, feature, position):
        """更新特征库中的对应ID特征和最后位置"""
        if feature is None or gid not in self.gallery: return
        
        # 更新特征 (移动平均)
        old_feat = self.gallery[gid]['feature']
        alpha = self.gallery_update_alpha
        new_feat = (1 - alpha) * old_feat + alpha * feature
        norm = np.linalg.norm(new_feat)
        self.gallery[gid]['feature'] = new_feat / norm if norm > 0 else new_feat
        
        # 更新位置
        self.gallery[gid]['last_position'] = position
    
    def mark_id_as_disembarked(self, global_id):
        """将指定的 global_id 在特征库中标记为“已下车”"""
        if global_id in self.gallery:
            self.gallery[global_id]['disembarked'] = True
            print(f"  [运动逻辑] ReID:{global_id} 已被标记为 '已下车', 不会再分配给车内人员。")

    def process_track(self, track_id, feature, current_position, current_frame_ids=None, is_inside_vehicle=True):
        """处理单个追踪轨迹，分配或更新ReID ID"""
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

        best_id, best_sim = self._find_best_match(
            mean_feature, current_position, 
            exclude_ids=current_frame_ids, 
            is_inside_vehicle=is_inside_vehicle
        )

        if best_id is not None and best_sim >= self.reid_threshold:
            buffer.assign_id(best_id)
            self._update_gallery(best_id, mean_feature, current_position)
        else:
            new_id = self.next_id
            self.gallery[new_id] = {
                'feature': mean_feature, 
                'disembarked': False,
                'last_position': current_position 
            }
            self.next_id += 1
            buffer.assign_id(new_id)
        
        return buffer.assigned_id

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
            max_spatial_distance=150 # 关键参数！根据视频分辨率和人物移动速度调整
        )
        print(f"  - ReID System initialized with max_spatial_distance = {self.reid_system.max_spatial_distance} pixels.")
        
        self.passenger_count = 0
        self.threshold_line_p1 = None
        self.threshold_line_p2 = None
        self.downside_ref_point = None
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

    def _get_side_of_line(self, point, line_p1, line_p2):
        return (line_p2[0] - line_p1[0]) * (point[1] - line_p1[1]) - (line_p2[1] - line_p1[1]) * (point[0] - line_p1[0])

    def is_in_downside_area(self, point):
        if self.threshold_line_p1 is None or self.downside_ref_point is None: return False
        ref_side = self._get_side_of_line(self.downside_ref_point, self.threshold_line_p1, self.threshold_line_p2)
        current_side = self._get_side_of_line(point, self.threshold_line_p1, self.threshold_line_p2)
        return (ref_side * current_side) > 0

    def run(self, video_path_str: str, save_output: bool = True):
        video_path = Path(video_path_str)
        if not video_path.exists():
            print(f"Error: 视频文件不存在 -> {video_path}")
            return
        print(f"\n{'='*60}\n处理视频: {video_path.name}\n{'='*60}")
        
        cap = cv2.VideoCapture(video_path_str)
        fps, w, h, total_frames = (int(cap.get(p)) for p in [cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT])
        
        ret_first, first_frame = cap.read()
        if not ret_first:
            print("Error: 无法读取视频的第一帧。"); cap.release(); return
        
        p1, p2, ref_point = visualize_select_line_and_side(first_frame)
        if p1 and p2 and ref_point:
            self.threshold_line_p1, self.threshold_line_p2, self.downside_ref_point = p1, p2, ref_point
            print(f"  - 门槛线及下车侧设置成功。")
        else:
            print("  - 未设置门槛线或下车侧，程序将退出。"); cap.release(); return

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        writer = None
        if save_output:
            output_dir = Path("single_video_output_reid_counting")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{video_path.stem}_final_optimized.mp4"
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

            cv2.line(vis_frame, self.threshold_line_p1, self.threshold_line_p2, (0, 0, 255), 3)
            cv2.circle(vis_frame, self.downside_ref_point, 7, (0, 0, 255), -1)
            cv2.putText(vis_frame, "Disembarkation Line", (self.threshold_line_p1[0], self.threshold_line_p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if boxes.id is not None:
                for box_coords, track_id in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
                    track_id = int(track_id)
                    x1, y1, x2, y2 = map(int, box_coords)
                    person_bottom_center = ((x1 + x2) // 2, y2)
                    feature = self._extract_feature(vis_frame, box_coords)
                    
                    is_currently_inside_vehicle = not self.is_in_downside_area(person_bottom_center)
                    
                    global_person_id = self.reid_system.process_track(
                        track_id, feature, person_bottom_center, 
                        current_frame_reid_ids, 
                        is_inside_vehicle=is_currently_inside_vehicle
                    )
                    
                    buffer = self.reid_system.track_buffers.get(track_id)
                    label, color = "...", (200, 200, 200)
                    
                    if global_person_id is not None:
                        current_frame_reid_ids.add(global_person_id)
                        color = get_color_for_id(global_person_id)
                        label = f"ID:{global_person_id}"
                        if buffer and buffer.is_locked(): label += " (L)"

                        if not buffer.has_counted:
                            in_downside = self.is_in_downside_area(person_bottom_center)
                            if in_downside:
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
                                    print(f"Frame {frame_idx}: Passenger Disembarked! ReID: {global_person_id}, Trk: {track_id}")
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
    print(f"\n{'='*60}\n单视频乘客检测、ReID与下车计数脚本 (最终优化版)\n{'='*60}")

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
    YOLO_MODEL = r"D:\AI\DeepSORT_YOLOv8_Pytorch\best1.pt" # 替换为您的YOLOv8模型路径
    REID_MODEL_PATH = r"D:\AI\DeepSORT_YOLOv8_Pytorch\detect-track-reid\final_osnet_ain_x1_0_12ids.pth" # 替换为您的ReID模型路径
    REID_NUM_CLASSES = 12 # 这个数字必须和你训练该ReID模型时使用的训练集ID数量完全一致

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
    print(f"\n{'='*60}\n所有处理完成！\n{'='*60}")

if __name__ == "__main__":
    main()