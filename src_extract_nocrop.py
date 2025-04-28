import os
import cv2
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch
import glob
from PIL import Image

# 初始化MTCNN模型用于人脸检测
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(margin=0, thresholds=[0.6, 0.7, 0.7], device=device)
    
def extract_face_from_video(video_path, person_id):
    """从视频中提取高质量人脸"""
    print(f"提取视频 {video_path} 中的人脸")
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return None
    
    # 获取视频信息
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 定义采样点, 遍历所有视频帧，步长为10
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
            best_face = frame
            best_face_size = face_size
    
    # 释放视频
    cap.release()
    print(f"best face size: {best_face_size}")

    return best_face, best_face_size

def main():
    # 设置路径
    data_dir = "data/Celeb-real"
    output_dir = "data/src_original"
    
    # 提取59个ID的人脸
    successful_extractions = 0
    
    for i in tqdm(range(59), desc="提取人脸"):
        person_id = f"id{i}"
        global_best_face = None
        global_best_face_size = 0
        # 遍历所有以person_id开头的视频
        video_path = os.path.join(data_dir, f"{person_id}_*.mp4")
        for video_path in glob.glob(video_path):
            if os.path.exists(video_path):
                best_face, best_face_size = extract_face_from_video(video_path, person_id)
            
                if best_face is not None and best_face_size > global_best_face_size:
                    global_best_face = best_face
                    global_best_face_size = best_face_size
            else:
                print(f"警告：找不到ID为 {person_id} 的任何视频")
        
        if global_best_face is not None:
            successful_extractions += 1
            print(f"person_id: {person_id}, global_best_face_size: {global_best_face_size}")
            # Save the best face
            # 获取 data/targ/ 中以 person_id 开头的所有文件名
            targ_pattern = f"data/targ/{person_id}-*.png"
            targ_files = glob.glob(targ_pattern)
            
            if targ_files:
                for targ_file in targ_files:
                    # 提取文件名，保持相同的命名格式
                    file_name = os.path.basename(targ_file)
                    face_path = os.path.join(output_dir, file_name)
                    cv2.imwrite(face_path, global_best_face)
            print(f"为 {person_id} 保存了 {len(targ_files)} 个人脸图像")
    
    print(f"完成! 成功提取了 {successful_extractions}/59 个ID的人脸图像")

if __name__ == "__main__":
    main()