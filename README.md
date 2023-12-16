# AnimateAnyone_unofficial
Unofficial implementation of Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation 

- Pre-trained model: stable diffusion 1.5
- Resolution: 512
- Batch size: 2
- GPU: single A6000 48G

- Trainging time: 12hours, global iteration: 37800
![](https://github.com/MingtaoGuo/AnimateAnyone_unofficial/blob/main/display/sd1.5_iter37800_bs2.png)

- Under training...

:smile::smile::rocket::rocket:__Due to the absence of official source code release, this unofficial code has not been thoroughly validated, and there are still many details to be verified. We welcome collaboration from the community to collectively implement and refine this algorithm！！！__


## Description   
--------------

This repo is mainly to re-implement AnimateAnyone based on official [ControlNet](https://github.com/lllyasviel/ControlNet) repository.
- AnimateAnyone: [Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation](https://arxiv.org/pdf/2311.17117.pdf)
## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN
- Python 3

### Installation
- Clone the repository:
``` 
git clone https://github.com/MingtaoGuo/AnimateAnyone_unofficial.git
cd AnimateAnyone_unofficial
```
- Dependencies:  
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment are provided in `environment.yaml`.

### First stage training
- Downloading the pre-trained stable diffusion [v1-5-pruned.ckpt
](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main).

- Extraction of CLIP Vision Embedder Weights
``` 
python tool_get_visionclip.py
```
- Copying Weights from Pretrained stable diffusion model to ReferenceNet
``` 
python tool_add_reference.py ./models/v1-5-pruned.ckpt ./models/reference_sd15_ini.ckpt
```
- Preprocessing Video Dataset (Video Decoding and Human Skeleton Extraction)
``` 
python tool_get_pose.py --mp4_path Dataset/fashion_mp4/ \
                        --save_frame_path Dataset/fashion_png/ \
                        --save_pose_path Dataset/fashion_pose/
```
Dataset Organization Structure
```
Dataset
  ├── fashion_mp4
      ├── 1.mp4
      ├── 2.mp4
       ...
  ├── fashion_png
      ├── 1.mp4
          ├── 1.png
          ├── 2.png
           ...
      ├── 2.mp4
          ├── 1.png
          ├── 2.png
             ...
         ...
  ├── fashion_pose
      ├── 1.mp4
          ├── 1.png
          ├── 2.png
           ...
      ├── 2.mp4
          ├── 1.png
          ├── 2.png
             ...
         ...
```       
- Train
```
python tutorial_train_animate.py
```
- Custom Dataset 
```
import json
import os 
import cv2
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, path="Dataset/"):
        self.path = path
        self.videos = os.listdir(path + "fashion_png")

    def __len__(self):
        return len(self.videos) * 10

    def __getitem__(self, idx):
        video_name = np.random.choice(self.videos)
        frames = np.random.choice(os.listdir(self.path + "/fashion_png/" + video_name), [2])
        ref_frame, tgt_frame = frames[0], frames[1]
        ref_bgr = cv2.imread(self.path + "/fashion_png/"  + video_name + "/" + ref_frame)
        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        ref_rgb = (ref_rgb.astype(np.float32) / 127.5) - 1.0

        tgt_bgr = cv2.imread(self.path + "/fashion_png/"  + video_name + "/" + tgt_frame)
        tgt_rgb = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB)
        tgt_rgb = (tgt_rgb.astype(np.float32) / 127.5) - 1.0

        skt_bgr = cv2.imread(self.path + "/fashion_pose/"  + video_name + "/" + tgt_frame)
        skt_rgb = cv2.cvtColor(skt_bgr, cv2.COLOR_BGR2RGB)
        skt_rgb = skt_rgb.astype(np.float32) / 255.0

        return dict(target=tgt_rgb, vision=ref_rgb, reference=ref_rgb, skeleton=skt_rgb)
```
## Author 
Mingtao Guo
E-mail: gmt798714378 at hotmail dot com
## Acknowledgement
We are very grateful for the official [ControlNet](https://github.com/lllyasviel/ControlNet) repository.
## Reference
[1]. Hu, Li, et al. "Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation." arXiv preprint arXiv:2311.17117 (2023).
