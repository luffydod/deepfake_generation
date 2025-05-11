# DeepFake-Generation-Celeb-DF-v2

本项目基于InsightFace库的预训练模型，实现了高质量的人脸换脸算法，同时集成了人脸增强功能。项目主要针对Celeb-DF v2数据集提供了完整的预处理工作流程，包括视频帧提取、人脸检测、换脸处理及后期优化等功能。通过简单的命令行操作，用户可以实现从原始视频到高质量换脸结果的全流程处理。项目既支持人脸区域裁剪处理，也支持保留原始分辨率的完整帧处理方式，以满足不同应用场景的需求。

（已弃用）based in [DiffFace](https://github.com/hxngiee/DiffFace.git)

based in [roop](https://github.com/s0md3v/roop/tree/main/roop)

## 环境依赖

```bash
pip install -r requirements.txt
```

## 运行方式

项目提供了一键运行脚本，可以直接执行以下命令：

```bash
bash roop_run.sh
```

## 命名规则

image_list.txt
****
Take id0_id1_0000_00060.png as an example. It stands for the 60th frame of the target video id0_0000.mp4 with its face swapped to id1.

raw_name: `{targ_id}_{src_id}_{video_index}_{frame_index}.png`

video_name: `{targ_id}_{video_index}.mp4`

中间结果中的命名规范如下。

process_name: `{src_id}-{targ_id}_{video_index}_{frame_index}.png`

## data process

（已弃用）根据 `data/image_list.txt` 要求，从 `data/Celeb-real` 中截取视频帧，作为目标图像，输出在 `data/targ` 目录。

```bash
python target_extract.py
```

（已弃用）提取要换取的面部图像，id{0-58}，文件命名根据targ目录保持同步，输出在 `data/src` 目录。

```bash
python src_extract.py
```

下面是无裁剪版本的数据处理脚本，这些脚本会提取完整的视频帧而不进行人脸裁剪，保留原始分辨率。这对于需要保留更多图像上下文信息的应用场景非常有用。

从视频中提取完整帧（不裁剪人脸区域）：

```bash
python target_extract_nocrop.py
```

提取源人物的完整图像（不裁剪人脸区域）：

```bash
python src_extract_nocrop.py
```

无裁剪版本的输出目录为：

- 目标图像：data/targ_original
- 源图像：data/src_original

