import os
import cv2
import torch
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from config import get_config
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    print("AMP is not available")

from torchvision.transforms import transforms

from utils import (
    get_heatmap_peak_coords,
    get_memory_format,
    get_head_mask
)
from models import get_model, load_pretrained
from datasets.transforms.ToColorMap import ToColorMap

class Inference:
    def __init__(self, MODEL_PATH):
        self.config = get_config()


        self.input_size = self.config.input_size
        self.output_size = self.config.output_size

        # self.is_test_set = is_test_set
        self.head_bbox_overflow_coeff = 0.1  # Will increase/decrease the bbox of the head by this value (%)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.depth_transform = transforms.Compose(
            [ToColorMap(plt.get_cmap("magma")), transforms.Resize((self.input_size, self.input_size)), transforms.ToTensor()]
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running on {self.device}")

        # Load model
        print("Loading model")
        model = get_model(self.config, device=self.device)


        # Do an evaluation or continue and prepare training
        # if config.eval_weights:
        print("Preparing evaluation")

        pretrained_dict = torch.load(MODEL_PATH, map_location=self.device)
        pretrained_dict = pretrained_dict.get("model_state_dict") or pretrained_dict.get("model")

        self.model = load_pretrained(model, pretrained_dict)
        self.model.eval()


    def run_overSingleFrame(self, img, depth, head_bbox):

        x_min = head_bbox[0]
        y_min = head_bbox[1]
        x_max = head_bbox[2]
        y_max = head_bbox[3]

        img = img.convert("RGB")
        img_cp = np.array(img.copy())
        width, height = img.size

        # Expand face bbox a bit
        x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
        x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
        y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        head = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)

        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Load depth image
        depth = depth.convert("L")

        # Apply transformation to images...
        if self.image_transform is not None:
            img = self.image_transform(img)
            face = self.image_transform(face)

        # ... and depth
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)

        img = img.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config)).unsqueeze(0)
        depth = depth.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config)).unsqueeze(0)
        face = face.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config)).unsqueeze(0)
        head = head.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config)).unsqueeze(0)

        gaze_heatmap_pred, _, _, _ = self.model(img, depth, head, face)

        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1).cpu()

        # gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1).cpu()
        pred_x, pred_y = get_heatmap_peak_coords(gaze_heatmap_pred[0])
        norm_p = torch.tensor([pred_x / float(self.config.output_size), pred_y / float(self.config.output_size)])

        converted = list(map(int, [x_min, y_min, x_max, y_max]))
        starting_point = ((converted[0] + converted[2]) // 2, (converted[1] + converted[3]) // 2)
        ending_point = (int(norm_p[0] * img_cp.shape[1]), int(norm_p[1] * img_cp.shape[0]))

        return starting_point, ending_point
    def makeBatch(self, img, depth, cmbHeadBox):
        head_list = []
        depth_list = []
        frame_list = []
        face_list =[]
        for iter, head_box in enumerate(cmbHeadBox):
            img_cp = img.copy()
            depth_cp = depth.copy()
            x_min = head_box[0]
            y_min = head_box[1]
            x_max = head_box[2]
            y_max = head_box[3]

            img_cp = img_cp.convert("RGB")
            width, height = img_cp.size

            # Expand face bbox a bit
            x_min -= self.head_bbox_overflow_coeff * abs(x_max - x_min)
            y_min -= self.head_bbox_overflow_coeff * abs(y_max - y_min)
            x_max += self.head_bbox_overflow_coeff * abs(x_max - x_min)
            y_max += self.head_bbox_overflow_coeff * abs(y_max - y_min)

            x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

            head = get_head_mask(x_min, y_min, x_max, y_max, width, height, resolution=self.input_size).unsqueeze(0)

            # Crop the face
            face = img_cp.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

            # Load depth image
            depth_cp = depth_cp.convert("L")

            # Apply transformation to images...
            if self.image_transform is not None:
                img_cp = self.image_transform(img_cp)
                face = self.image_transform(face)

            # ... and depth
            if self.depth_transform is not None:
                depth_cp = self.depth_transform(depth_cp)
            head_list.append(head)
            depth_list.append(depth_cp)
            frame_list.append(img_cp)
            face_list.append(face)



        return head_list, frame_list, depth_list, face_list



    def runOverBatch(self, img, depth, cmbHeadBox):
        img_cp = np.array(img.copy())
        head, img, depth, face = self.makeBatch(img, depth, cmbHeadBox)
        
        head = torch.stack(head)
        img = torch.stack(img)
        depth = torch.stack(depth)
        face = torch.stack(face)


        img = img.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config))
        depth = depth.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config))
        face = face.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config))
        head = head.to(self.device, non_blocking=True, memory_format=get_memory_format(self.config))

        gaze_heatmap_pred, _, _, _ = self.model(img, depth, head, face)

        gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1).cpu()
        return_pnts = []
        for gaze, head_box in zip(gaze_heatmap_pred, cmbHeadBox):
            pred_x, pred_y = get_heatmap_peak_coords(gaze)
            norm_p = torch.tensor([pred_x / float(self.config.output_size), pred_y / float(self.config.output_size)])
    
            converted = list(map(int, head_box))
            starting_point = ((converted[0] + converted[2]) // 2, (converted[1] + converted[3]) // 2)
            ending_point = (int(norm_p[0] * img_cp.shape[1]), int(norm_p[1] * img_cp.shape[0]))
            return_pnts.append([starting_point, ending_point])
        return return_pnts


if __name__ == "__main__":

    # Make runs repeatable as much as possible
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    load_dotenv()
    MODEL_PATH = './output/spatial_depth_late_fusion_gazefollow_gazefollow/default/ckpt_epoch_41.pth'
    obj = Inference(MODEL_PATH)


    IMAGE_DIR = '../../dataset/elm_gaze/images/'
    csv_path = '../../dataset/elm_gaze/all_heads.txt'

    column_names = [
        "frame",
        "left",
        "top",
        "right",
        "bottom"]

    df = pd.read_csv(csv_path, sep=",", names=column_names, usecols=column_names, index_col=False)
    df_grpd = df.reset_index().groupby(['frame'])
    for ind, row in df_grpd:
        cmb_head_bbox = []
        for d in row.index.tolist():
            cmb_head_bbox.append([row.loc[d, 'left'],
                                  row.loc[d, 'top'],
                                  row.loc[d, 'right'],
                                  row.loc[d, 'bottom']])

        frame_raw = Image.open(os.path.join(IMAGE_DIR, ind))
        depth_path = IMAGE_DIR.replace("images", "depth")
        depth = Image.open(os.path.join(depth_path, ind))
        return_pnts = obj.runOverBatch(frame_raw, depth, cmb_head_bbox)
        img_cp = np.array(frame_raw)

        for starting_point, ending_point in return_pnts:
            # print(starting_point, ending_point)
            if starting_point != None and ending_point != None:
                # r = random.randint(0, 255)
                # g = random.randint(0, 255)
                # b = random.randint(0, 255)
                b = 255;
                g = 0;
                r = 0
                cv2.line(img_cp, tuple(map(int, starting_point)), tuple(map(int, ending_point)), (b, g, r), 5)
                cv2.circle(img_cp, tuple(map(int, ending_point)), 15,
                           (b, g, r), -1)
        plt.imshow(img_cp, cmap='gray')
        plt.show()

    print('All images successfully completed !!!!!')