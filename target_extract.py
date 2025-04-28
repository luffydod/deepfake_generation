import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN
from PIL import Image

def extract_frames_from_video(src_id, video_name, data_dir, output_dir, frame_indices=None, target_size=256, scale=1.3):
    """
    从视频中提取指定帧，进行人脸裁切，并保存为指定分辨率
    
    Args:
        src_id: 要换取的人物脸型ID
        video_name: 视频名称
        data_dir: 数据目录
        output_dir: 输出目录
        frame_indices: 要提取的帧索引列表，如果为None则提取所有帧
        target_size: 输出图像的分辨率
        scale: 人脸扩展比例，控制裁切区域大小
    
    Returns:
        提取的帧数量
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    video_path = os.path.join(data_dir, f"{video_name}.mp4")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return 0
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 如果没有指定帧索引，则提取所有帧
    if frame_indices is None:
        frame_indices = range(total_frames)
    
    # 确保所有帧索引有效
    frame_indices = [i for i in frame_indices if 0 <= i < total_frames]
    
    # 初始化MTCNN模型用于人脸检测
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(margin=0, thresholds=[0.6, 0.7, 0.7], device=device)
    
    extracted_count = 0
    
    # 提取指定帧
    for frame_idx in frame_indices:
        # 设置当前帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # 读取当前帧
        ret, frame = cap.read()
        if not ret:
            print(f"警告：无法读取视频 {video_path} 的第 {frame_idx} 帧")
            continue
        
        # 转换为RGB用于人脸检测
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # 检测人脸
        boxes, probs, points = mtcnn.detect(pil_img, landmarks=True)
        
        # 如果未检测到人脸，跳过当前帧
        if boxes is None or len(boxes) == 0:
            print(f"警告：在视频 {video_path} 的第 {frame_idx} 帧中未检测到人脸")
            continue
        
        # 选择概率最高的人脸
        best_idx = np.argmax(probs)
        box = boxes[best_idx]
        
        # 扩展人脸边界框
        xmin, ymin, xmax, ymax = [int(b) for b in box]
        w = xmax - xmin
        h = ymax - ymin
        
        # 计算扩展后的正方形边界框
        size_bb = int(max(w, h) * scale)
        center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
        
        # 边界检查，确保裁剪区域在图像内
        xmin = max(int(center_x - size_bb // 2), 0)
        ymin = max(int(center_y - size_bb // 2), 0)
        
        # 检查裁剪区域大小是否超出图像边界
        size_bb = min(frame.shape[1] - xmin, size_bb)
        size_bb = min(frame.shape[0] - ymin, size_bb)
        
        # 裁剪人脸
        face_crop = frame[ymin:ymin + size_bb, xmin:xmin + size_bb]
        
        # 调整裁剪后的人脸大小为目标尺寸
        face_resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        # 保存帧图像
        frame_filename = f"{src_id}-{video_name}_{frame_idx:05d}.png"
        save_path = os.path.join(output_dir, frame_filename)
        
        cv2.imwrite(save_path, face_resized)
        
        # 保存原始帧和坐标信息
        # 创建记录目录
        record_dir = os.path.join("data/record", f"{src_id}-{video_name}_{frame_idx:05d}")
        os.makedirs(record_dir, exist_ok=True)
        # 保存原始帧
        raw_path = os.path.join(record_dir, "raw.png")
        cv2.imwrite(raw_path, frame)
        
        # 保存坐标和缩放信息
        pos_path = os.path.join(record_dir, "pos.txt")
        with open(pos_path, 'w') as f:
            f.write(f"xmin={xmin}\n")
            f.write(f"ymin={ymin}\n")
            f.write(f"size_bb={size_bb}\n")
            f.write(f"target_size={target_size}\n")
            f.write(f"scale={scale}\n")
            f.write(f"original_width={frame.shape[1]}\n")
            f.write(f"original_height={frame.shape[0]}\n")
        
        extracted_count += 1
    
    # 释放视频对象
    cap.release()
    
    return extracted_count

def parse_image_list(image_list_path):
    """
    解析image_list.txt文件，提取需要的帧信息
    
    Args:
        image_list_path: image_list.txt的路径
    
    Returns:
        包含视频ID和帧索引的字典
    """
    frames_to_extract = {}
    
    with open(image_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 解析文件名格式：id0_id1_0000_00060.png
            parts = line.split('_')
            if len(parts) != 4:
                print(f"警告：无法解析文件名 {line}")
                continue
            
            # 获取目标视频ID和帧索引
            targ_id = parts[0]
            src_id = parts[1]
            video_seq = parts[2]
            frame_idx = int(parts[3].split('.')[0])
            
            # 构建视频名称
            video_name = f"{src_id}-{targ_id}_{video_seq}"
            
            # 添加到字典
            if video_name not in frames_to_extract:
                frames_to_extract[video_name] = []
            
            frames_to_extract[video_name].append(frame_idx)
    
    return frames_to_extract

def main():
    # 设置路径
    data_dir = "data/Celeb-real"  # 原始视频目录
    output_dir = "data/targ"  # 输出目录
    image_list_path = "data/image_list.txt"  # 需要提取的帧列表
    target_size = 128  # 输出分辨率
    
    # 解析image_list.txt
    print("解析image_list.txt...")
    frames_to_extract = parse_image_list(image_list_path)
    
    if not frames_to_extract:
        print("错误：未找到需要提取的帧")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取每个视频的指定帧
    total_extracted = 0
    for video_name, frame_indices in tqdm(frames_to_extract.items(), desc="提取视频帧"):
        src_id = video_name.split('-')[0]
        v_name = video_name.split('-')[1]
        # 提取指定帧
        extracted = extract_frames_from_video(
            src_id, 
            v_name, 
            data_dir, 
            output_dir, 
            frame_indices, 
            target_size=target_size, 
            scale=1.3
        )
        total_extracted += extracted
        
        print(f"已从 {v_name}.mp4 提取 {extracted} 帧")
    
    print(f"总共提取了 {total_extracted} 帧图像")

if __name__ == "__main__":
    main()