import cv2
import os 
import argparse

import numpy as np
import torch

from tqdm import tqdm 
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector


apply_openpose = OpenposeDetector()


def openpose_detect(input_image, detect_resolution=512, image_resolution=512):
    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map, _ = apply_openpose(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

    return detected_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp4_path", type=str, required=True, help="Folder with *.mp4 files for training the model.")
    parser.add_argument("--save_frame_path", type=str, required=True)
    parser.add_argument("--save_pose_path", type=str, required=True)
    args = parser.parse_args()

    mp4_path = args.mp4_path
    save_frame_path = args.save_frame_path
    save_pose_path = args.save_pose_path

    if not os.path.exists(save_frame_path):
        os.mkdir(save_frame_path)

    if not os.path.exists(save_pose_path):
        os.mkdir(save_pose_path)

    # ------------ Decoding mp4 to png ------------- #
    files = os.listdir(mp4_path)
    for file in files:
        os.mkdir(save_frame_path + file)
        os.system(f"ffmpeg -i {mp4_path}/{file} {save_frame_path}/{file}/%1d.png")

    # ------------ Normalizing each frame to 512 x 512 ------------- #
    files = os.listdir(save_frame_path)
    for file in tqdm(files):
        imgnames = os.listdir(save_frame_path + file)
        for imgname in imgnames:
            img = cv2.imread(save_frame_path + file + "/" + imgname)
            h, w = img.shape[0], img.shape[1]
            new_h = 512
            new_w = int(new_h / h * w)
            resized_img = cv2.resize(img, (new_w, new_h))
            canvas = np.ones([512, 512, 3]) * 255
            start_x = int((512 - new_w) / 2)
            end_x = int(start_x + new_w)
            canvas[:, start_x:end_x] = resized_img
            cv2.imwrite(save_frame_path + file + "/" + imgname, canvas)

    # ------------ Detecting pose maps for each frame ------------- #
    files = os.listdir(save_frame_path)
    for file in tqdm(files):
        os.mkdir(save_pose_path + file)
        imgnames = os.listdir(save_frame_path + file)
        for imgname in imgnames:
            img = cv2.imread(save_frame_path + file + "/" + imgname)
            pose_map = openpose_detect(img)
            cv2.imwrite(save_pose_path + file + "/" + imgname , pose_map)

