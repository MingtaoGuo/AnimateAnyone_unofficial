import json
import os 
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, path="/mnt/gmt/Dataset/"):
        self.path = path
        self.videos = os.listdir(path + "fashion_png")

    def __len__(self):
        return len(self.videos) * 10

    def __getitem__(self, idx):
        video_name = np.random.choice(self.videos)
        frames = np.random.choice(os.listdir(self.path + "/fashion_png/" + video_name), [2])
        ref_frame, tgt_frame = frames[0], frames[1]
        ref_bgr = cv2.imread(self.path + "/fashion_png/"  + video_name + "/" + ref_frame)
        # ref_bgr = cv2.resize(ref_bgr, (256, 256))
        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        ref_rgb = (ref_rgb.astype(np.float32) / 127.5) - 1.0

        tgt_bgr = cv2.imread(self.path + "/fashion_png/"  + video_name + "/" + tgt_frame)
        # tgt_bgr = cv2.resize(tgt_bgr, (256, 256))
        tgt_rgb = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2RGB)
        tgt_rgb = (tgt_rgb.astype(np.float32) / 127.5) - 1.0

        skt_bgr = cv2.imread(self.path + "/fashion_pose/"  + video_name + "/" + tgt_frame)
        # skt_bgr = cv2.resize(skt_bgr, (256, 256))
        skt_rgb = cv2.cvtColor(skt_bgr, cv2.COLOR_BGR2RGB)
        skt_rgb = skt_rgb.astype(np.float32) / 255.0

        return dict(target=tgt_rgb, vision=ref_rgb, reference=ref_rgb, skeleton=skt_rgb)

