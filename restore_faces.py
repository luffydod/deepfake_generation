import os
import cv2
import numpy as np
from tqdm import tqdm
import glob

def parse_pos_file(pos_path):
    """
    解析位置文件，获取坐标和缩放信息
    
    Args:
        pos_path: 位置信息文件路径
    
    Returns:
        包含所有位置参数的字典
    """
    pos_info = {}
    with open(pos_path, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            if key in ['xmin', 'ymin', 'size_bb', 'target_size', 'original_width', 'original_height']:
                pos_info[key] = int(value)
            else:
                pos_info[key] = float(value)
    return pos_info

def restore_face_to_frame(processed_face_path, record_base_dir, output_dir):
    """
    将处理后的人脸图像还原到原始视频帧
    
    Args:
        processed_face_path: 处理后的人脸图像路径
        record_base_dir: 记录信息基础目录
        output_dir: 输出目录
    
    Returns:
        是否成功
    """
    # 提取文件名信息
    face_filename = os.path.basename(processed_face_path)
    file_name = os.path.splitext(face_filename)[0]  # 去掉.png后缀
    
    # 构建记录目录路径
    record_dir = os.path.join(record_base_dir, file_name)
    
    # 检查记录目录是否存在
    if not os.path.exists(record_dir):
        print(f"警告：未找到记录目录 {record_dir}")
        return False
    
    # 读取原始帧
    raw_path = os.path.join(record_dir, "raw.png")
    if not os.path.exists(raw_path):
        print(f"警告：未找到原始帧 {raw_path}")
        return False
    
    original_frame = cv2.imread(raw_path)
    
    # 读取位置信息
    pos_path = os.path.join(record_dir, "pos.txt")
    if not os.path.exists(pos_path):
        print(f"警告：未找到位置信息 {pos_path}")
        return False
    
    pos_info = parse_pos_file(pos_path)
    
    # 读取处理后的人脸图像
    processed_face = cv2.imread(processed_face_path)
    
    # 调整大小以匹配原始人脸区域
    resized_face = cv2.resize(processed_face, (pos_info['size_bb'], pos_info['size_bb']), interpolation=cv2.INTER_LANCZOS4)
    
    # 将人脸放回原始帧
    xmin, ymin = pos_info['xmin'], pos_info['ymin']
    
    # 确保不超出边界
    h, w, _ = original_frame.shape
    if xmin + pos_info['size_bb'] > w:
        print(f"警告：人脸区域超出右边界，进行裁剪")
        resized_face = resized_face[:, :w-xmin]
    
    if ymin + pos_info['size_bb'] > h:
        print(f"警告：人脸区域超出下边界，进行裁剪")
        resized_face = resized_face[:h-ymin, :]
    
    # 创建人脸区域的掩码（简单圆形掩码，可根据需要调整）
    mask = np.zeros((pos_info['size_bb'], pos_info['size_bb']), dtype=np.uint8)
    center = pos_info['size_bb'] // 2
    radius = int(pos_info['size_bb'] * 0.45)  # 使用稍小的半径进行平滑混合
    cv2.circle(mask, (center, center), radius, 255, -1)
    
    # 创建过渡区域的边缘模糊效果
    mask = cv2.GaussianBlur(mask, (41, 41), 0)
    
    # 将掩码扩展为3通道
    mask_3channel = cv2.merge([mask, mask, mask]) / 255.0
    
    # 复制原始帧
    result_frame = original_frame.copy()
    
    # 提取要替换的区域
    roi = result_frame[ymin:ymin+resized_face.shape[0], xmin:xmin+resized_face.shape[1]]
    
    # 使用掩码混合原始区域和新人脸
    blended = roi * (1 - mask_3channel[:roi.shape[0], :roi.shape[1]]) + resized_face[:roi.shape[0], :roi.shape[1]] * mask_3channel[:roi.shape[0], :roi.shape[1]]
    
    # 将混合结果放回原始帧
    result_frame[ymin:ymin+resized_face.shape[0], xmin:xmin+resized_face.shape[1]] = blended
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, face_filename)
    cv2.imwrite(output_path, result_frame)
    
    return True

def main():
    # 设置路径
    processed_dir = "output_extract"  # 处理后的人脸图像目录
    record_base_dir = "data/record"  # 记录信息根目录
    output_dir = "output_restore"  # 输出目录
    
    # 查找所有处理后的人脸图像
    processed_faces = glob.glob(os.path.join(processed_dir, "*.png"))
    
    if not processed_faces:
        print("错误：未找到处理后的人脸图像")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个人脸图像
    success_count = 0
    total_count = len(processed_faces)
    
    for face_path in tqdm(processed_faces, desc="还原人脸"):
        # 还原人脸到原始帧
        if restore_face_to_frame(face_path, record_base_dir, output_dir):
            success_count += 1
    
    print(f"总共处理 {total_count} 个人脸图像，成功还原 {success_count} 个")

if __name__ == "__main__":
    main()