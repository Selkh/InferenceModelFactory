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


class MMdetectionItem(Item):
    def __init__(self, data, metas, img_id):
        self.data = data
        self.metas = metas
        self.img_id = img_id
        self.batch_masks = None



class MMdetectionModel(OnnxModel):
    def __init__(self):
        super(MMdetectionModel, self).__init__()
        self.options = self.get_options()
        self.options.add_argument("--model_path",
                            default="model/solo_r50_1x-mmdet-op13-fp32.onnx",
                            help="Onnx path")
        self.options.add_argument('--data_path',
                                  default='./coco',
                                  type=str,
                                  help='dataset path')
        self.options.add_argument('--scale',
                                  default=(800, 1333),
                                  type=int,
                                  nargs='+',
                                  help='model input image scale')
        self.options.add_argument('--mean',
                                  default=(123.675, 116.28, 103.53),
                                  type=float,
                                  nargs='+',
                                  help='image normalize mean value')
        self.options.add_argument('--std',
                                  type=float,
                                  default=(58.395, 57.12, 57.375),
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
        self.options.add_argument("--batch_size",
                                  type=int,
                                  default=1,
                                  help="batch_size")


    # TODO : common func
    def __imresize(self, item):
        """
        Resize image
        """
        img = item.data
        h, w = img.shape[:2]
        scale = self.options.get_scale()
        new_h, new_w = scale
        resized_img = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        w_scale = scale[1] / w
        h_scale = scale[0] / h
        scale_factor = np.array(
            [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

        item.data = resized_img
        item.metas['img_shape'] = resized_img.shape
        # in case that there is no padding
        item.metas['pad_shape'] = resized_img.shape
        item.metas['scale_factor'] = scale_factor
        item.metas['keep_ratio'] = self.options.get_keep_ratio()

    def __imrescale(self, item):
        """
        Rescale image
        """
        img = item.data
        h, w = img.shape[:2]
        scale = self.options.get_scale()
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
        new_size = int(w * float(scale_factor) +
                       0.5), int(h * float(scale_factor) + 0.5)
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        new_h, new_w = resized_img.shape[:2]
        w_scale = new_w / w
        h_scale = new_h / h
        scale_factor = np.array(
            [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

        item.data = resized_img
        item.metas['img_shape'] = resized_img.shape
        # in case that there is no padding
        item.metas['pad_shape'] = resized_img.shape
        item.metas['scale_factor'] = scale_factor
        item.metas['keep_ratio'] = self.options.get_keep_ratio()

    def __normalize(self, item):
        """
        Normalize image
        """
        img = item.data
        img = img.copy().astype(np.float32)
        mean = np.float64(np.array(self.options.get_mean(),
                                   dtype=np.float32).reshape(1, -1))
        stdinv = 1 / \
            np.float64(np.array(self.options.get_std(),
                                dtype=np.float32).reshape(1, -1))
        if self.options.get_to_rgb():
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        item.data = img

    def __pad(self, item, pad_val=0):
        """
        Padding image
        """
        img = item.data
        pad_h = int(
            np.ceil(img.shape[0] / self.options.get_size_divisor())) * self.options.get_size_divisor()
        pad_w = int(
            np.ceil(img.shape[1] / self.options.get_size_divisor())) * self.options.get_size_divisor()
        padding = (0, 0, pad_w - img.shape[1], pad_h - img.shape[0])
        img = cv2.copyMakeBorder(img,
                                 padding[1],
                                 padding[3],
                                 padding[0],
                                 padding[2],
                                 cv2.BORDER_CONSTANT,
                                 value=pad_val)
        item.data = img
        item.metas['pad_shape'] = img.shape
        item.metas['pad_fixed_size'] = None
        item.metas['pad_size_divisor'] = self.options.get_size_divisor()

    def __image2tensor(self, item):
        """
        Image to tensor
        """
        img = item.data
        img = np.transpose(img, axes=(2, 0, 1))
        img = np.expand_dims(img, axis=0)
        item.data = img

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
        return MMdetectionItem(img, metas, img_id)

    
    def preprocess(self, item):
        max_h = 0
        max_w = 0
        if self.options.get_keep_ratio():
            self.__imrescale(item)
        else:
            self.__imresize(item)
        self.__normalize(item)
        self.__pad(item)
        self.__image2tensor(item)
        pad_h, pad_w = item.metas['pad_shape'][:2]
        if pad_h > max_h:
            max_h = pad_h
        if pad_w > max_w:
            max_w = pad_w

        pad_h, pad_w = item.metas['pad_shape'][:2]
        new_img = np.zeros([1, 3, max_h, max_w], dtype=np.float32)
        new_img[:, :, :pad_h, :pad_w] = item.data

        item.data = np.squeeze(new_img, axis=(0,))
        item.metas['ori_shape'] = np.array(
            item.metas['ori_shape'], dtype=np.int64)
        item.metas['pad_shape'] = np.array(
            item.metas['pad_shape'], dtype=np.int64)
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

        outputs = sess.run(output_names, input_feed)



        for z in zip(items, outputs[0]):
            Model.assignment(*z, 'dets')
        for z in zip(items, outputs[1]):
            Model.assignment(*z, 'labels')
        if len( outputs) == 3:
            for z in zip(items, outputs[2]):
                Model.assignment(*z, 'batch_masks')

        return items

    
    def postprocess(self, item):

        num_classes = self.options.get_num_classes()
        de = []
        post_rescale = True
        if 'mask' in self.options.get_model_path():
                post_rescale = False

        
        dets, labels = item.dets.copy(), item.labels.copy()


        if post_rescale:
            scale_factor = item.metas['scale_factor']
            dets[:, :4] /= scale_factor
        if dets.shape[0] == 0:
            dets_results = [np.zeros((0, 5), dtype=np.float32)
                            for _ in range(num_classes)]
        else:
            dets_results = [dets[labels == cls_id, :]
                            for cls_id in range(num_classes)]
        if item.batch_masks is not None:
            img_h, img_w = item.metas['img_shape'][:2]
            ori_h, ori_w = item.metas['ori_shape'][:2]
            mask_results = []
            for cls_id in range(num_classes):
                masks = item.batch_masks[labels == cls_id, :]
                if masks.size == 0:
                    cls_mask_results = masks
                else:
                    cls_mask_results = []
                    for mask in masks:
                        if post_rescale:
                            cls_mask = cv2.resize(
                                mask[:img_h, :img_w], (ori_w, ori_h))
                            cls_mask = cls_mask > 0.5
                            cls_mask = cls_mask.astype(np.bool_)
                            cls_mask = np.expand_dims(cls_mask, axis=0)
                        else:
                            cls_mask = np.expand_dims(mask, axis=0)
                        cls_mask_results.append(cls_mask)
                    cls_mask_results = np.concatenate(
                        cls_mask_results, axis=0)
                mask_results.append(cls_mask_results)
            dets_results = (dets_results, mask_results)
        de.append(dets_results)

        results = []
        for det in de:
            if isinstance(det, tuple):
                all_bboxes, all_masks = det
            else:
                all_bboxes, all_masks = det, None
            for label, bboxes in enumerate(all_bboxes):
                if bboxes.size == 0:
                    continue
                for i in range(0, bboxes.shape[0]):
                    bbox = bboxes[i]
                    xmin = float(bbox[0])
                    ymin = float(bbox[1])
                    xmax = float(bbox[2])
                    ymax = float(bbox[3])
                    score = float(bbox[4])
                    result = {'image_id': item.img_id,
                              'category_id': CLASSMAP[label],
                              'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                              'score': score}
                    if all_masks is not None:
                        mask = all_masks[label][i]
                        mask = mask_util.encode(np.array(mask[:, :, np.newaxis],
                                                         order='F', dtype='uint8'))[0]
                        result['segmentation'] = mask
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