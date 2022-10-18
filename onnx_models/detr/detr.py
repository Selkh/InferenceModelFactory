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
from PIL import Image
import cv2
import numpy as np
# import copy
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


class DetrFactory(OnnxModelFactory):
    model = "detr"

    def new_model():
        return Detr()


class DetrItem(Item):
    def __init__(self, data, metas, img_id):
        self.data = data
        self.metas = metas
        self.img_id = img_id


class Detr(OnnxModel):
    def __init__(self):
        super(Detr, self).__init__()
        self.options = self.get_options()
        self.options.add_argument('--model_path',
                                  default='./model/detr-r50-op13-fp32-H-W.onnx',
                                  help='onnx path')
        self.options.add_argument('--data_path',
                                  default='./data',
                                  type=str,
                                  help='dataset path')
        self.options.add_argument('--mean',
                                  default=(0.485, 0.456, 0.406),
                                  type=float,
                                  nargs='+',
                                  help='image normalize mean value')
        self.options.add_argument('--std',
                                  type=float,
                                  default=(0.229, 0.224, 0.225),
                                  nargs='+',
                                  help='image normalize std value')
        self.options.add_argument('--to_rgb',
                                  type=bool,
                                  default=True,
                                  help='whether convert image to rgb channel')
        self.options.add_argument('--size_divisor',
                                  type=int,
                                  default=32,
                                  help='padding size divisor')
        self.options.add_argument('--num_classes',
                                  type=int,
                                  default=80,
                                  help='class number')

        # TODO : args reset
        """
        if 'caffe' in args.model:
            args.mean = [103.530, 116.280, 123.675]
            args.std = [1.0, 1.0, 1.0]
            args.to_rgb = False
        args.batch_size = 1
        """

    # TODO : common func
    def get_size_with_aspect_ratio(self, image_size, size, max_size=None):
            w, h = image_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)

    def get_size(self, image_size, size, max_size=None):
            if isinstance(size, (list, tuple)):
                return size[::-1]
            else:
                return self.get_size_with_aspect_ratio(image_size, size, max_size)

    def __imresize(self, item):
        """
        Resize image
        """
        img = item.data
        shape = item.metas['img_shape'][:2]
        # img = img.convert('RGB')
        size = self.get_size(shape, size=800, max_size=1333)
        resized_img = img.resize(size[::-1], 2)
        resized_img = np.array(resized_img)


        item.data = resized_img
        item.metas['img_shape'] = resized_img.shape


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
            img = Image.open(img_name).convert("RGB")

        metas['img_shape'] = img.size[:2]
        metas['ori_shape'] = img.size[:2]
        metas['img_fields'] = ['img']
        return DetrItem(img, metas, img_id)

    def preprocess(self, item):
        
        self.__imresize(item)
        
        h0, w0 =  item.metas['ori_shape'][0:2]
        h, w = item.metas['img_shape'][0:2]

        shapes = np.array([w0, h0])
        img = item.data

        # Convert
        img = img.transpose(2,0,1)
        img = img.astype(np.float32)
        img /= 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        mean = mean.reshape(-1,1,1)
        std = std.reshape(-1,1,1)
        img = (img-mean)/std
        img = np.ascontiguousarray(img)


        max_size = [3,1336,1336]
        batch_shape = [1] + max_size
        
        b, c, h, w = batch_shape
        imgs = np.zeros((c, h, w))
        masks = np.ones((b, c, h, w))

        imgs[: img.shape[0], : img.shape[1], : img.shape[2]] = img
        masks[..., : img.shape[1], :img.shape[2]] = False

        imgs = imgs.astype(np.float32)
        masks = masks.astype(np.float32)


        item.data = imgs
        item.metas['masks'] = masks
        item.metas['ori_shape'] = np.array(
            item.metas['ori_shape'], dtype=np.int64)
        item.metas['img_shape'] = np.array(
            item.metas['img_shape'], dtype=np.int64)

        return item

    def run_internal(self, sess, items):
        datas = Model.make_batch([item.data for item in items])
        masks = items[0].metas['masks'].copy()
        inputs = {
            'images': datas,
            'masks': masks
        }
        
        items[0].outputs = sess.run(None, inputs)
        return items

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x[:,:,0], x[:,:,1], x[:,:,2], x[:,:,3]
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return np.stack(b, axis=-1)

    def softmax(self, x,axis=-1):

        means = np.mean(x, axis, keepdims=True)[0]
        x_exp = np.exp(x-means)
        x_exp_sum = np.sum(x_exp, axis, keepdims=True)

        return x_exp/x_exp_sum

    def convert_to_xywh(self, boxes):
        xmin, ymin, xmax, ymax = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return np.stack((xmin, ymin, xmax - xmin, ymax - ymin), axis=1)

    def postprocess(self, item):
        

        batch_size = 1
        num_classes = self.options.get_num_classes()
        shapes =  item.metas['ori_shape'][0:2]
        out_logits, out_bbox = item.outputs[0], item.outputs[1]   

        prob = self.softmax(out_logits,-1)
        scores, labels = np.max(prob[...,:-1],axis=-1), np.argmax(prob[...,:-1],axis=-1)
        boxes = self.box_cxcywh_to_xyxy(out_bbox)
        
        img_w, img_h = shapes[0], shapes[1]
        scale_fct = np.stack([img_w, img_h, img_w, img_h], axis=0)
        boxes = boxes * scale_fct
      

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        jdict = []
        for sj, result in enumerate(results):
            scores = result['scores']
            labels = result['labels']
            boxes = result['boxes']

            for k, (score,label,box) in enumerate(zip(scores, labels, boxes)):
                dic = {'image_id': item.img_id ,
                                'category_id': label,
                                'bbox': [round(x, 3) for x in box],
                                'score': round(score, 5)}
                jdict.append(dic)
                

        return jdict

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