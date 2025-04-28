# DeepFake-Generation-Celeb-DF-v2

based in [DiffFace](https://github.com/hxngiee/DiffFace.git)

## 命名规则

image_list.txt

Take id0_id1_0000_00060.png as an example. It stands for the 60th frame of the target video id0_0000.mp4 with its face swapped to id1.

raw_name: `{targ_id}_{src_id}_{video_index}_{frame_index}.png`

video_name: `{targ_id}_{video_index}.mp4`

中间结果中的命名规范如下。

process_name: `{src_id}-{targ_id}_{video_index}_{frame_index}.png`

## requirements

- dlib
- face_alignment
- kornia
- lpips

```bash
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 -c pytorch

pip install opencv-python==4.6.0.66 \
    lpips==0.1.4 \
    face_alignment==1.3.5 \
    kornia==0.6.7 \
    matplotlib

```

## data process

根据 `data/image_list.txt` 要求，从 `data/Celeb-real` 中截取视频帧，作为目标图像，输出在 `data/targ` 目录。

```bash
    python target_extract.py
```

提取要换取的面部图像，id{0-58}，文件命名根据targ目录保持同步，输出在 `data/src` 目录。

```bash
    python src_extract.py
```
