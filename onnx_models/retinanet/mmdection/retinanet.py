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


class RetinaNetFactory(OnnxModelFactory):
    model = "retinanet"

    def new_model():
        return RetinaNet()


class RetinaNetItem(Item):
    def __init__(self, data, metas, img_id):
        self.data = data
        self.metas = metas
        self.img_id = img_id


class RetinaNet(OnnxModel):
    def __init__(self):
        super(RetinaNet, self).__init__()
        self.options = self.get_options()
        self.options.add_argument('--model_path',
                                  default='./model/retinanet-r101-mmdet-pt-op13-fp32.onnx',
                                  help='onnx path')
        self.options.add_argument('--data_path',
                                  default='coco/',
                                  type=str,
                                  help='dataset path')
        self.options.add_argument('--scale',
                                  default=(800, 1216),
                                  type=int,
                                  nargs='+',
                                  help='model input image scale')
        self.options.add_argument('--mean',
                                  default=(103.530, 116.280, 123.675),
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
        self.options.add_argument('--threshold',
                                  type=int,
                                  default=0.05,
                                  help='score threshold')

        # TODO : args reset
        """
        if 'caffe' in args.model:
            args.mean = [103.530, 116.280, 123.675]
            args.std = [1.0, 1.0, 1.0]
            args.to_rgb = False
        args.batch_size = 1
        """


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
        return RetinaNetItem(img, metas, img_id)
    

    def preprocess(self, item):
        '''
            __imrescale is for mask_rcnn etc. which need to keep the original scale, may be not
            necessary for other detection models. 
            The developer can check the onnx graphs for special models to justify the needed input
        '''
        img = item.data
        h, w, _ = img.shape
        scale = min(800 / h, 1216 / w)
        resized = cv2.resize(img, dsize = None, fx = scale, fy = scale)
        h, w, _ = resized.shape
        padded = np.pad(resized, [[0,800 - h],[0,1216 - w],[0,0]])
        normalized = (padded - np.array([103.530, 116.280, 123.675]))
        inputs = np.expand_dims(normalized, axis = 0).astype(np.float32)
        inputs = np.transpose(inputs, (0,3,1,2))

        item.data = np.squeeze(inputs, axis=(0,))
        item.metas['scale'] = scale
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

        # f=open('fix_log.txt',"a")
        # f.write(str(input_feed['input'].shape) + '\n')
        # f.close

       
        
        items[0].output = sess.run(output_names, input_feed)
        return items

    def postprocess(self, item):
        dets, labels = item.output

        batch_dets = np.squeeze(dets, axis = 0)
        batch_labels = np.squeeze(labels, axis = 0)
        batch_dets, batch_scores = batch_dets[:,:4].copy(), batch_dets[:,4].copy()
        scale = item.metas['scale']
        batch_dets /= scale

        batch_size = 1
        results = []

        num_classes = self.options.get_num_classes()
        for i in range(batch_size):
            dets, labels, scores = batch_dets, batch_labels, batch_scores
            
            for box, score, label in zip(dets, scores, labels):
                if score > self.options.get_threshold():
                    f=open('fix_log.txt',"a")
                    f.write(str(box) + '  ' + str(score) + '  ' + str(label) + '\n')
                    f.close
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