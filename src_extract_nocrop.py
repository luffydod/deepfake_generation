import os
import cv2
import numpy as np
from tqdm import tqdm
import glob
import torch
from facenet_pytorch import MTCNN
from PIL import Image

def extract_face_from_video(video_path, output_dir, person_id, target_size=256, scale=1.3):
    """从视频中提取高质量人脸"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return None
    
    # 获取视频信息
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 定义采样点 - 均匀地从视频中选择10个时间点
    sample_points = np.linspace(0, frame_count-1, 10, dtype=int)
    
    # 初始化MTCNN模型用于人脸检测
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(margin=0, thresholds=[0.6, 0.7, 0.7], device=device)
    
    best_face = None
    best_face_size = 0
    
    # 采样并检测人脸
    for frame_idx in sample_points:
        # 设置帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 转换为RGB用于人脸检测
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # 检测人脸
        boxes, probs, points = mtcnn.detect(pil_img, landmarks=True)
        
        # 如果未检测到人脸，跳过当前帧
        if boxes is None or len(boxes) == 0:
            continue
        
        # 选择概率最高的人脸
        best_idx = np.argmax(probs)
        box = boxes[best_idx]
        
        # 获取人脸边界框
        xmin, ymin, xmax, ymax = [int(b) for b in box]
        
        # 计算人脸大小
        face_size = (xmax - xmin) * (ymax - ymin)
        
        # 如果这个人脸比之前找到的更大，就更新
        if face_size > best_face_size:
            # 计算扩展后的正方形边界框
            w = xmax - xmin
            h = ymax - ymin
            size_bb = int(max(w, h) * scale)
            center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
            
            # 边界检查，确保裁剪区域在图像内
            xmin = max(int(center_x - size_bb // 2), 0)
            ymin = max(int(center_y - size_bb // 2), 0)
            
            # 检查裁剪区域大小是否超出图像边界
            size_bb = min(frame.shape[1] - xmin, size_bb)
            size_bb = min(frame.shape[0] - ymin, size_bb)
            
            # 裁剪人脸
            best_face = frame[ymin:ymin + size_bb, xmin:xmin + size_bb]
            best_face_size = face_size
    
    # 释放视频
    cap.release()
    
    # 如果找到了人脸，保存它
    if best_face is not None:
        # # 应用简单的图像增强
        # best_face_lab = cv2.cvtColor(best_face, cv2.COLOR_BGR2LAB)
        # l, a, b = cv2.split(best_face_lab)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # cl = clahe.apply(l)
        # enhanced_lab = cv2.merge((cl, a, b))
        # enhanced_face = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # # 调整图像大小为目标尺寸
        # resized_face = cv2.resize(enhanced_face, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        resized_face = best_face
        
        # 获取 data/targ/ 中以 person_id 开头的所有文件名
        targ_pattern = f"data/targ/{person_id}-*.png"
        targ_files = glob.glob(targ_pattern)
        
        if targ_files:
            saved_files = []
            for targ_file in targ_files:
                # 提取文件名，保持相同的命名格式
                file_name = os.path.basename(targ_file)
                face_path = os.path.join(output_dir, file_name)
                cv2.imwrite(face_path, resized_face)
                saved_files.append(face_path)
            return saved_files
        else:
            # 如果没有找到匹配的源文件，使用默认命名方式
            print(f"警告：未找到与 {targ_pattern} 匹配的源文件，使用默认命名")
            # face_path = os.path.join(output_dir, f"{person_id}.png")
            # cv2.imwrite(face_path, resized_face)
            # return [face_path]
    
    return None

def main():
    # 设置路径
    data_dir = "data/Celeb-real"
    output_dir = "data/src_original"
    target_size = 256  # 输出分辨率
    
    # 提取59个ID的人脸
    successful_extractions = 0
    
    for i in tqdm(range(59), desc="提取人脸"):
        person_id = f"id{i}"
        
        # 随机选择一个视频序号 (0000-0009)
        video_seq = f"{np.random.randint(0, 10):04d}"
        video_path = os.path.join(data_dir, f"{person_id}_{video_seq}.mp4")
        
        # 如果随机选择的视频不存在，尝试其他视频
        if not os.path.exists(video_path):
            for seq in range(10):
                alt_video_path = os.path.join(data_dir, f"{person_id}_{seq:04d}.mp4")
                if os.path.exists(alt_video_path):
                    video_path = alt_video_path
                    break
        
        if not os.path.exists(video_path):
            print(f"警告：找不到ID为 {person_id} 的任何视频")
            continue
        
        results = extract_face_from_video(video_path, output_dir, person_id, target_size=target_size, scale=1.3)
        
        # 如果第一个视频没有找到好的人脸，尝试其他视频
        if not results:
            print(f"在视频 {video_path} 中没有找到好的人脸，尝试其他视频...")
            for seq in range(10):
                alt_video_path = os.path.join(data_dir, f"{person_id}_{seq:04d}.mp4")
                if alt_video_path != video_path and os.path.exists(alt_video_path):
                    results = extract_face_from_video(alt_video_path, output_dir, person_id, target_size=target_size, scale=1.3)
                    if results:
                        print(f"在备选视频 {alt_video_path} 中找到了好的人脸")
                        break
        
        if results:
            successful_extractions += 1
            print(f"为 {person_id} 保存了 {len(results)} 个人脸图像")
    
    print(f"完成! 成功提取了 {successful_extractions}/59 个ID的人脸图像")

if __name__ == "__main__":
    main()