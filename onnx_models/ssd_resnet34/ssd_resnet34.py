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


class Ssd_Resnet34Factory(OnnxModelFactory):
    model = "ssd_resnet34"

    def new_model():
        return Ssd_Resnet34()


class Ssd_Resnet34Item(Item):
    def __init__(self, data, metas, img_id):
        self.data = data
        self.metas = metas
        self.img_id = img_id


class Ssd_Resnet34(OnnxModel):
    def __init__(self):
        super(Ssd_Resnet34, self).__init__()
        self.options = self.get_options()
        self.options.add_argument('--model_path',
                                  default='./model/ssd_resnet34_1200x1200_wo_nms-pt-op13-fp32.onnx',
                                  help='onnx path')
        self.options.add_argument('--framework',
                                  default='pt',
                                  type=str,
                                  help='model framework')
        self.options.add_argument('--data_path',
                                  default='coco/',
                                  type=str,
                                  help='dataset path')
        self.options.add_argument('--scale',
                                  default=(1200, 1200),
                                  type=int,
                                  nargs='+',
                                  help='model input image scale')
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


    # TODO : common func
    def __imresize(self, item):
        """
        Resize image
        """
        scale=self.options.get_scale()
        img = item.data
        h, w = img.shape[:2]
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
        scale=self.options.get_scale()
        img = item.data
        h, w = img.shape[:2]
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
        # pad_h = self.options.get_scale()[0]
        # pad_w = self.options.get_scale()[1]

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
        return Ssd_Resnet34Item(img, metas, img_id)

    def preprocess(self, item):
        '''
            __imrescale is for mask_rcnn etc. which need to keep the original scale, may be not
            necessary for other detection models. 
            The developer can check the onnx graphs for special models to justify the needed input
        '''
        # if self.options.get_keep_ratio():
        #     self.__imrescale(item)
        # else:
        self.__imresize(item)
        self.__normalize(item)

        self.__image2tensor(item)

        item.data = np.squeeze(item.data, axis=(0,))
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
            flag = 0
            for name in input_names:
                print("name:" + name + " " + str(len(input_names)))
                if name not in inputs:
                    inputs[name] = []
                if name == 'input' or  name == 'image':
                    inputs[name].append(datas)
                    inputs[name] = np.concatenate(inputs[name], axis=0)
                else:
                    inputs[name].append(items[0].metas[name])

            input_feed = {}
            for name in input_names:
                    input_feed[name] = inputs[name]
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

    def postprocess(self, item):

        batch_size = 1
        results = []

        num_classes = self.options.get_num_classes()
        for i in range(batch_size):
            raw_dets = item.output
            f=open('fix_log.txt',"a")
            f.write(str(raw_dets[0].shape) + '  ' + str(raw_dets[1].shape)  + '\n')
            f.close
            if self.options.get_framework() == 'pt':
                dets = raw_dets[0][0]
                labels = raw_dets[1][0]
                scores = raw_dets[2][0]
            else:
                dets = raw_dets[1][0]
                labels = raw_dets[3][0]
                scores = raw_dets[2][0]
            
            for box, score, label in zip(dets, scores, labels):
                if score > self.options.get_threshold():
                    # f=open('fix_log.txt',"a")
                    # f.write(str(box) + '  ' + str(score) + '  ' + str(label) + '\n')
                    # f.close
                    xmin = float(box[0])
                    ymin = float(box[1])
                    xmax = float(box[2])
                    ymax = float(box[3])
                    result = {'image_id': item.img_id,
                                'category_id': CLASSMAP[label],
                                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                                'score': float(score)}
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