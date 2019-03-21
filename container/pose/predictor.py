import argparse

import cv2
import numpy as np
import torch
import flask
import json
import time
import skimage

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints, BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS
from modules.load_state import load_state
from val import normalize, pad_width




class Pose(object):
    net = None

    def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
                   pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
        height, width, _ = img.shape
        scale = net_input_height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if not cpu:
            tensor_img = tensor_img.cuda()

        stages_output = net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad

    @classmethod
    def predict(cls,img, height_size, cpu, stride, upsample_ratio):
        net = cls.get_net()

        net = net.eval()
        if not cpu:
            net = net.cuda()

        heatmaps, pafs, scale, pad = Pose.infer_fast(net, img, height_size, stride, upsample_ratio, cpu)
        return heatmaps, pafs, scale, pad

    @classmethod
    def get_net(cls):
        if cls.net is None:
            cls.net = PoseEstimationWithMobileNet()
            checkpoint = torch.load('checkpoint_iter_370000.pth.tar', map_location='cpu')
            load_state(cls.net, checkpoint)
        return cls.net

# The flask app for serving predictions
app = flask.Flask(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@app.route('/ping', methods=['GET'])
def ping():

    health = Pose.get_net() is not None  # You can insert a health check here
    #MaskRCNNService.get_model()
    status = 200  if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():

    #img_stringIO = BytesIO(flask.request.data)
    #image = skimage.io.imread(img_stringIO)

    image = skimage.io.imread('./bdawk.jpg')

    # Do the prediction

    stride = 8
    upsample_ratio = 4
    color = [0, 224, 255]
    orig_img = image.copy()

    #start = time.time()

    heatmaps, pafs, scale, pad  = Pose.predict( image, 256, True, stride=stride, upsample_ratio = upsample_ratio)
    #end = time.time()-start

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(18):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

    #display_image(pose_entries, all_keypoints, scale, pad,image,orig_img)

    #json_result = json.dumps({'prediction':{'heatmaps':heatmaps,'pafs':pafs,'class_ids':scale}}, cls=NumpyEncoder)
    json_result = json.dumps({'prediction':{'pose_entries':pose_entries,'all_keypoints':all_keypoints,'pad':pad,'scale':scale}}, cls=NumpyEncoder)
    #print(json_result)


    return flask.Response(response=json_result, status=200, mimetype='application/json')


def display_image(pose_entries, all_keypoints, scale, pad,img,orig_img):
    stride = 8
    upsample_ratio = 4
    color = [0, 224, 255]

    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = pose_entries[n][kpt_a_id]
            if global_kpt_a_id != -1:
                x_a, y_a = all_keypoints[int(global_kpt_a_id), 0:2]
                cv2.circle(img, (int(x_a), int(y_a)), 3, color, -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = pose_entries[n][kpt_b_id]
            if global_kpt_b_id != -1:
                x_b, y_b = all_keypoints[int(global_kpt_b_id), 0:2]
                cv2.circle(img, (int(x_b), int(y_b)), 3, color, -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), color, 2)

    img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
    #cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
    cv2.imwrite('out2.jpg',img)


print('DGRABS - Starting service HERE WE GO')

if __name__== "__main__":
    transformation()