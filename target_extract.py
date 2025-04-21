import os
import cv2
import numpy as np
import face_recognition
from tqdm import tqdm
import glob

def extract_face_from_video(video_path, output_dir, person_id):
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
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 定义采样点 - 均匀地从视频中选择10个时间点
    sample_points = np.linspace(0, frame_count-1, 10, dtype=int)
    
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
        
        # 检测人脸
        face_locations = face_recognition.face_locations(frame)
        if not face_locations:
            continue
        
        # 选择最大的人脸
        face_areas = [(right-left)*(bottom-top) for top, right, bottom, left in face_locations]
        largest_face_idx = np.argmax(face_areas)
        top, right, bottom, left = face_locations[largest_face_idx]
        
        # 计算人脸大小
        face_size = (right-left)*(bottom-top)
        
        # 如果这个人脸比之前找到的更大，就更新
        if face_size > best_face_size:
            # 扩大人脸区域以包含更多上下文
            height, width = frame.shape[:2]
            margin = int((right - left) * 0.3)  # 30% 边距
            
            extended_top = max(0, top - margin)
            extended_bottom = min(height, bottom + margin)
            extended_left = max(0, left - margin)
            extended_right = min(width, right + margin)
            
            best_face = frame[extended_top:extended_bottom, extended_left:extended_right]
            best_face_size = face_size
    
    # 释放视频
    cap.release()
    
    # 如果找到了人脸，保存它
    if best_face is not None:
        # 应用简单的图像增强
        best_face_lab = cv2.cvtColor(best_face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(best_face_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_face = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 获取 data/src/ 中以 person_id 开头的所有文件名
        src_pattern = f"data/src/{person_id}-*.png"
        src_files = glob.glob(src_pattern)
        
        if src_files:
            saved_files = []
            for src_file in src_files:
                # 提取文件名，保持相同的命名格式
                file_name = os.path.basename(src_file)
                face_path = os.path.join(output_dir, file_name)
                cv2.imwrite(face_path, enhanced_face)
                saved_files.append(face_path)
            return saved_files
        else:
            # 如果没有找到匹配的源文件，使用默认命名方式
            print(f"警告：未找到与 {src_pattern} 匹配的源文件，使用默认命名")
            face_path = os.path.join(output_dir, f"{person_id}.png")
            cv2.imwrite(face_path, enhanced_face)
            return [face_path]
    
    return None

def main():
    # 设置路径
    data_dir = "data/Celeb-real"
    output_dir = "data/targ"
    
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
        
        results = extract_face_from_video(video_path, output_dir, person_id)
        
        # 如果第一个视频没有找到好的人脸，尝试其他视频
        if not results:
            print(f"在视频 {video_path} 中没有找到好的人脸，尝试其他视频...")
            for seq in range(10):
                alt_video_path = os.path.join(data_dir, f"{person_id}_{seq:04d}.mp4")
                if alt_video_path != video_path and os.path.exists(alt_video_path):
                    results = extract_face_from_video(alt_video_path, output_dir, person_id)
                    if results:
                        print(f"在备选视频 {alt_video_path} 中找到了好的人脸")
                        break
        
        if results:
            successful_extractions += 1
            print(f"为 {person_id} 保存了 {len(results)} 个人脸图像")
    
    print(f"完成! 成功提取了 {successful_extractions}/59 个ID的人脸图像")

if __name__ == "__main__":
    main()