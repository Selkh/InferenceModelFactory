"""
/* Copyright 2018 The Enflame Tech Company. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
"""
# !/usr/bin/python
# -*- coding: utf-8 -*-

from onnx_models.base import OnnxModelFactory, OnnxModel
from common.dataset import read_dataset, Item
import cv2
import numpy as np
import torch
import torch.nn as nn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util
from common.model import Model


CLASSMAP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]


class CenterNetFactory(OnnxModelFactory):
    model = "centernet"

    def new_model():
        return CenterNet()


class CenterNetItem(Item):
    def __init__(self, data, metas, img_id):
        self.data = data
        self.metas = metas
        self.img_id = img_id


class CenterNet(OnnxModel):
    def __init__(self):
        super(CenterNet, self).__init__()
        self.options = self.get_options()
        self.options.add_argument('--model_path',
                                  default='./model/CenterNet_dlav0-pt-op13-fp32-N.onnx',
                                  help='onnx path')
        self.options.add_argument('--data_path',
                                  default='coco/',
                                  type=str,
                                  help='dataset path')
        self.options.add_argument('--scale',
                                  default=(512, 512),
                                  type=int,
                                  nargs='+',
                                  help='model input image scale')
        self.options.add_argument('--mean',
                                  default=(0.408, 0.447, 0.470),
                                  type=float,
                                  nargs='+',
                                  help='image normalize mean value')
        self.options.add_argument('--std',
                                  type=float,
                                  default=(0.289, 0.274, 0.278),
                                  nargs='+',
                                  help='image normalize std value')
        self.options.add_argument('--to_rgb',
                                  type=bool,
                                  default=True,
                                  help='whether convert image to rgb channel')
        self.options.add_argument('--keep_ratio',
                                  type=bool,
                                  default=True,
                                  help='whether keep ratio when resize image')
        self.options.add_argument('--size_divisor',
                                  type=int,
                                  default=32,
                                  help='padding size divisor')
        self.options.add_argument('--num_classes',
                                  type=int,
                                  default=80,
                                  help='class number')
        self.options.add_argument('--threshold',
                                  type=int,
                                  default=0.05,
                                  help='score threshold')
        self.options.add_argument("--conf-thres",
                                  default=0.005,
                                  help="confidence threshold")
        self.options.add_argument("--iou-thres",
                                  default=0.45,
                                  help="NMS Iou threshold")

        # TODO : args reset

    def create_dataset(self):
        self.anno = COCO(
            '{}/annotations/instances_val2017.json'.format(self.options.get_data_path()))
        return read_dataset(self.anno.getImgIds())

    def load_data(self, img_id):
        img_info = self.anno.loadImgs([img_id])[0]
        img_name = '{}/val2017/{}'.format(
            self.options.get_data_path(), img_info['file_name'])
        metas = dict()
        metas['filename'] = None
        if isinstance(img_name, str):
            metas['filename'] = img_name
            with open(img_name, 'rb') as f:
                img_buf = f.read()
            img_np = np.frombuffer(img_buf, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        metas['img_shape'] = img.shape
        metas['ori_shape'] = img.shape
        metas['img_fields'] = ['img']
        return CenterNetItem(img, metas, img_id)

    def preprocess(self, item):
        '''
            __imrescale is for mask_rcnn etc. which need to keep the original scale, may be not
            necessary for other detection models. 
            The developer can check the onnx graphs for special models to justify the needed input
        '''
        scale = self.options.get_scale()
        mean = self.options.get_mean()
        std = self.options.get_std()
        resized_image = cv2.resize(item.data, (scale[1], scale[0]))
        inp_image = ((resized_image / 255. - mean) /
                     std).astype(np.float32)

        inputs = inp_image.transpose(2, 0, 1).reshape(
            1, 3, scale[0], scale[1])

        item.data = np.squeeze(inputs, axis=(0,))
        item.metas['ori_shape'] = np.array(
            item.metas['ori_shape'], dtype=np.int64)
        item.metas['img_shape'] = np.array(
            item.metas['img_shape'], dtype=np.int64)

        return item

    def run_internal(self, sess, items):
        datas = Model.make_batch([item.data for item in items])

        # TODO : common func
        def get_input_feed():
            """
            Feed input tensors to model inputs
            """
            inputs = {}
            for name in input_names:
                print("name:", name)
                if name not in inputs:
                    inputs[name] = []
                if name == 'input':
                    inputs[name].append(datas)
                else:
                    inputs[name].append(items[0].metas[name])
            inputs['input'] = np.concatenate(inputs['input'], axis=0)

            input_feed = {}
            for name in input_names:
                if name == 'input':
                    input_feed[name] = inputs[name]
                else:
                    input_feed[name] = inputs[name][0]
            return input_feed

        def get_output_names():
            """
            Get model output tensor names
            """
            output_names = []
            for node in sess.get_outputs():
                output_names.append(node.name)
            return output_names

        def get_input_names():
            """
            Get model input tensor names
            """
            input_names = []
            for node in sess.get_inputs():
                input_names.append(node.name)
            return input_names

        output_names = get_output_names()
        input_names = get_input_names()
        input_feed = get_input_feed()

        items[0].output = sess.run(output_names, input_feed)

        return items

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _topk(self, scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / K).int()
        topk_inds = self._gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(
            batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(
            batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def ctdet_decode(self, heat, wh, reg=None, cat_spec_wh=False, K=100):
        batch, cat, height, width = heat.size()

        # perform nms on heatmaps
        heat = self._nms(heat)

        scores, inds, clses, ys, xs = self._topk(heat, K=K)
        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = self._transpose_and_gather_feat(wh, inds)
        if cat_spec_wh:
            wh = wh.view(batch, K, cat, 2)
            clses_ind = clses.view(batch, K, 1, 1).expand(
                batch, K, 1, 2).long()
            wh = wh.gather(2, clses_ind).view(batch, K, 2)
        else:
            wh = wh.view(batch, K, 2)
        clses = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        detections = torch.cat([bboxes, scores, clses], dim=2)

        return detections

    def ctdet_post_process(self, dets, h, w, num_classes):
        # dets: batch x max_dets x dim
        # return 1-based class det dict
        ret = []
        down_ratio = 4
        for i in range(dets.shape[0]):
            top_preds = {}
            dets[i, :, 0] = dets[i, :, 0] * down_ratio * \
                w / self.options.get_scale()[1]
            dets[i, :, 1] = dets[i, :, 1] * down_ratio * \
                h / self.options.get_scale()[0]
            dets[i, :, 2] = dets[i, :, 2] * down_ratio * \
                w / self.options.get_scale()[1]
            dets[i, :, 3] = dets[i, :, 3] * down_ratio * \
                h / self.options.get_scale()[0]
            classes = dets[i, :, -1]
            for j in range(num_classes):
                inds = (classes == j)
                top_preds[j + 1] = np.concatenate([
                    dets[i, inds, :4].astype(np.float32),
                    dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
            ret.append(top_preds)
        return ret

    def to_float(self, x):
        return float("{:.2f}".format(x))

    def postprocess(self, item):
        batch_size = 1
        results = []
        num_classes = self.options.get_num_classes()

        for i in range(batch_size):
            hm = torch.from_numpy(item.output[0]).sigmoid_()
            wh = torch.from_numpy(item.output[1])
            reg = torch.from_numpy(item.output[2])

            dets = self.ctdet_decode(hm, wh, reg=reg, cat_spec_wh=False, K=100)
            dets = dets.numpy()

            dets = dets.reshape(1, -1, dets.shape[2])
            dets = self.ctdet_post_process(
                dets.copy(), item.metas['ori_shape'][0], item.metas['ori_shape'][1], num_classes)
            for j in range(1, num_classes + 1):
                dets[0][j] = np.array(
                    dets[0][j], dtype=np.float32).reshape(-1, 5)

            if dets[0] is not None:
                for cls_ind in dets[0]:
                    for box in dets[0][cls_ind]:
                        score = box[4]
                        if score >= 0.05:
                            box[2] -= box[0]
                            box[3] -= box[1]
                            bbox_out = list(map(self.to_float, box[0:4]))
                            result = {'image_id': item.img_id,
                                      'category_id': CLASSMAP[cls_ind-1],
                                      'bbox': bbox_out,
                                      'score': np.round(score, 5).tolist()}
                            # f=open('fix_log.txt',"a")
                            # f.write(str(CLASSMAP[cls_ind-1]) + ' ' + str(bbox_out) + ' ' +str(np.round(score, 5).tolist()) + '\n')
                            # f.close
                            results.append(result)
        return results

    def eval(self, collections):
        results = sum(collections, [])
        for iou_type in ['bbox', 'segm']:
            cocoGt = self.anno
            cocoDt = self.anno.loadRes(results)
            imgIds = sorted(cocoGt.getImgIds())
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
        return {}
