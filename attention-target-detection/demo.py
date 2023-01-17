import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from cv2 import resize as imresize
from model import ModelSpatial
from utils import imutils, evaluation

import cv2, random

class Inference:
    def __init__(self, model_weights, out_threshold = 200):
        self.input_resolution = 224
        self.output_resolution = 64
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.input_resolution, self.input_resolution)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.model = ModelSpatial()
        model_dict = self.model.state_dict()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(model_weights, map_location=self.device)
        pretrained_dict = pretrained_dict['model']
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        if self.device == 'cuda':
            self.model.cuda()
        self.model.train(False)

        self.out_threshold = out_threshold


    def runOverSingle(self,frame_raw, head_box):


        with torch.no_grad():

            frame_raw = frame_raw.convert('RGB')
            width, height = frame_raw.size
            head = frame_raw.crop((head_box)) # head crop

            head = self.image_transform(head) # transform inputs
            frame = self.image_transform(frame_raw)
            head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
                                                        resolution=self.input_resolution).unsqueeze(0)
            if self.device == 'cuda':
                head = head.unsqueeze(0).cuda()
                frame = frame.unsqueeze(0).cuda()
                head_channel = head_channel.unsqueeze(0).cuda()
            else:
                head = head.unsqueeze(0)
                frame = frame.unsqueeze(0)
                head_channel = head_channel.unsqueeze(0)

            # forward pass
            raw_hm, _, inout = self.model(frame, head_channel, head)

            # heatmap modulation
            raw_hm = raw_hm.cpu().detach().numpy() * 255
            raw_hm = raw_hm.squeeze()
            inout = inout.cpu().detach().numpy()
            inout = 1 / (1 + np.exp(-inout))
            inout = (1 - inout) * 255

            # # MyVis
            head_box = list(map(int, head_box))
            frame_raw = np.array(frame_raw)
            cv2.rectangle(frame_raw, (head_box[0], head_box[1]), (head_box[2], head_box[3]), (255, 0, 0), 2)
            if inout <  self.out_threshold:  # in-frame gaze
                pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                norm_p = [pred_x / self.output_resolution, pred_y / self.output_resolution]


                starting_point = ((head_box[0] + head_box[2]) // 2, (head_box[1] + head_box[3]) // 2)
                ending_point = (int(norm_p[0] * width), int(norm_p[1] * height))
                return starting_point, ending_point
            return None, None
    def makeBatch(self, frame_raw, cmbHeadBox, width, height):
        head_list = []
        head_channel_list = []
        frame_list = []
        for iter, head_box in enumerate(cmbHeadBox):
            head = frame_raw.crop((head_box))  # head crop

            head = self.image_transform(head)  # transform inputs

            frame = self.image_transform(frame_raw)

            head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
                                                        resolution=self.input_resolution).unsqueeze(0)
            head_list.append(head)
            head_channel_list.append(head_channel)
            frame_list.append(frame)
        return head_list, frame_list, head_channel_list
    def runOverBatch(self,frame_raw, cmbHeadBox):


        with torch.no_grad():

            frame_raw = frame_raw.convert('RGB')
            width, height = frame_raw.size

            head, frame,head_channel = self.makeBatch(frame_raw, cmbHeadBox, width, height)
            head = torch.stack(head)
            frame = torch.stack(frame)
            head_channel = torch.stack(head_channel)

            # head = frame_raw.crop((head_box)) # head crop

            # head = self.image_transform(head) # transform inputs

            # head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
            #                                             resolution=self.input_resolution).unsqueeze(0)
            if self.device == 'cuda':
                head = head.cuda()
                frame = frame.cuda()
                head_channel = head_channel.cuda()
            else:
                head = head
                frame = frame
                head_channel = head_channel

            # forward pass
            raw_hms, _, inouts = self.model(frame, head_channel, head)

            return_pnts = []
            for raw_hm, inout, head_box in zip(raw_hms, inouts, cmbHeadBox):
                # heatmap modulation
                raw_hm = raw_hm.cpu().detach().numpy() * 255
                raw_hm = raw_hm.squeeze()
                inout = inout.cpu().detach().numpy()
                inout = 1 / (1 + np.exp(-inout))
                inout = (1 - inout) * 255

                # # MyVis
                # head_box = list(map(int, head_box))
                # frame_raw = np.array(frame_raw)
                # cv2.rectangle(frame_raw, (head_box[0], head_box[1]), (head_box[2], head_box[3]), (255, 0, 0), 2)
                if inout <  self.out_threshold:  # in-frame gaze
                    pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                    norm_p = [pred_x / self.output_resolution, pred_y / self.output_resolution]
                    starting_point = ((head_box[0] + head_box[2]) // 2, (head_box[1] + head_box[3]) // 2)
                    ending_point = (int(norm_p[0] * width), int(norm_p[1] * height))
                    return_pnts.append([starting_point, ending_point])
                else:
                    return_pnts.append([None, None])
            return return_pnts

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_weights', type=str, help='model weights', default='trained_model/epoch_18_weights.pt')
    # parser.add_argument('--image_dir', type=str, help='images', default='data/gaze_follow_test/frames')
    # parser.add_argument('--head', type=str, help='head bounding boxes', default='data/gaze_follow_test/test.txt')
    # args = parser.parse_args()
    MODEL_WEIGHTS = 'trained_model/epoch_18_weights.pt'
    IMAGE_DIR = 'data/elm_gaze/frames'
    HEAD = 'data/elm_gaze/test.txt'

    obj = Inference(MODEL_WEIGHTS)

    column_names = ['frame', 'left', 'top', 'right', 'bottom']
    df = pd.read_csv(HEAD, names=column_names, index_col=0)
    df['left'] -= (df['right'] - df['left']) * 0.1
    df['right'] += (df['right'] - df['left']) * 0.1
    df['top'] -= (df['bottom'] - df['top']) * 0.1
    df['bottom'] += (df['bottom'] - df['top']) * 0.1

    df_grpd = df.reset_index().groupby(['frame'])

    for ind, row in df_grpd:
        cmb_head_bbox = []
        for d in row.index.tolist():
            cmb_head_bbox.append([row.loc[d, 'left'],
                  row.loc[d, 'top'],
                  row.loc[d, 'right'],
                  row.loc[d, 'bottom']])


        frame_raw = Image.open(os.path.join(IMAGE_DIR, ind))
        return_pnts = obj.runOverBatch(frame_raw, cmb_head_bbox)
        img_cp = np.array(frame_raw)

        for starting_point, ending_point in return_pnts:
            # print(starting_point, ending_point)
            if starting_point != None and ending_point != None:
                # r = random.randint(0, 255)
                # g = random.randint(0, 255)
                # b = random.randint(0, 255)
                b= 255; g = 0; r = 0
                cv2.line(img_cp, tuple(map(int,starting_point)), tuple(map(int,ending_point)), (b, g, r), 5)
                cv2.circle(img_cp, tuple(map(int,ending_point)), 15,
                           (b, g, r), -1)
        screen = cv2.cvtColor(img_cp, cv2.COLOR_RGB2BGR)
        cv2.imshow('Frame', screen)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    print('DONE!')

