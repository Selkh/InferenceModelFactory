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
from common.model import Model
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# solo_r50_fpn_8gpu_1x.py
test_cfg = dict(
    nms_pre=500,
    score_thr=0.1,
    mask_thr=0.5,
    update_thr=0.05,
    kernel='gaussian',  # gaussian/linear
    sigma=2.0,
    max_per_img=100)


class SOLOFactory(OnnxModelFactory):
    model = "solo"

    def new_model():
        return SOLO()


class SOLOItem(Item):
    def __init__(self, data,metas,img_id):
        self.data = data
        self.metas = metas
        self.img_id = img_id


class SOLO(OnnxModel):
    def __init__(self):
        super(SOLO, self).__init__()
        self.options = self.get_options()
        self.options.add_argument("--model_path",
                            default="model/solo_r50_1x-mmdet-op13-fp32.onnx",
                            help="Onnx path")
        self.options.add_argument('--data_path',
                            default='coco/',
                            type=str,
                            help='dataset path')
        self.options.add_argument('--device',
                            default='dtu',
                            help='dtu, gpu, cpu')
        self.options.add_argument('--config',
                            help='test config file path')
        self.options.add_argument('--out_file',
                            default="results_solo_segm.json",
                            help='output result file')
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
        self.options.add_argument('--batch_size',
                            type=int,
                            default=1,
                            help='batch size')


    def __imresize(self, item, scale = (800, 1333)):
        """
        Resize image
        """
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
        item.metas['ori_shape'] = img.shape
        item.metas['img_shape'] = resized_img.shape
        # in case that there is no padding
        item.metas['pad_shape'] = resized_img.shape
        item.metas['scale_factor'] = scale_factor
        item.metas['keep_ratio'] = self.options.get_keep_ratio()


    def __imrescale(self, item, scale = (800, 1333)):
        """
        Rescale image
        """
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
        mean = np.float64(np.array((123.675, 116.28, 103.53), dtype=np.float32).reshape(1, -1))
        stdinv = 1 / np.float64(np.array((58.395, 57.12, 57.375), dtype=np.float32).reshape(1, -1))
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


    @staticmethod
    def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
        """Matrix NMS for multi-class masks.

        Args:
            seg_masks (Tensor): shape (n, h, w)
            cate_labels (Tensor): shape (n), mask labels in descending order
            cate_scores (Tensor): shape (n), mask scores in descending order
            kernel (str):  'linear' or 'gauss'
            sigma (float): std in gaussian method
            sum_masks (Tensor): The sum of seg_masks

        Returns:
            Tensor: cate_scores_update, tensors of shape (n)
        """
        n_samples = len(cate_labels)
        if n_samples == 0:
            return []
        if sum_masks is None:
            sum_masks = seg_masks.sum((1, 2)).float()
        seg_masks = seg_masks.reshape(n_samples, -1).float()
        # inter.
        inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
        # union.
        sum_masks_x = sum_masks.expand(n_samples, n_samples)
        # iou.
        iou_matrix = (inter_matrix / (sum_masks_x +
                                      sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
        # label_specific matrix.
        cate_labels_x = cate_labels.expand(n_samples, n_samples)
        label_matrix = (cate_labels_x == cate_labels_x.transpose(
            1, 0)).float().triu(diagonal=1)
        # IoU compensation
        compensate_iou, _ = (iou_matrix * label_matrix).max(0)
        compensate_iou = compensate_iou.expand(
            n_samples, n_samples).transpose(1, 0)
        # IoU decay
        decay_iou = iou_matrix * label_matrix
        # matrix nms
        if kernel == 'gaussian':
            decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
            compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
            decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
        elif kernel == 'linear':
            decay_matrix = (1-decay_iou)/(1-compensate_iou)
            decay_coefficient, _ = decay_matrix.min(0)
        else:
            raise NotImplementedError
        # update the score.
        cate_scores_update = cate_scores * decay_coefficient
        return cate_scores_update


    def __get_seg_single(self, cate_preds,
                        seg_preds,
                        featmap_size,
                        img_shape,
                        ori_shape):

        assert len(cate_preds) == len(seg_preds)

        seg_num_grids = [40, 36, 24, 16, 12]
        strides_solo = [8, 8, 16, 32, 32]

        h, w, _ = img_shape
        upsampled_size_out = (featmap_size[0] * 4, featmap_size[1] * 4)

        # process.
        inds = (cate_preds > test_cfg['score_thr'])
        # category scores.
        cate_scores = cate_preds[inds]
        if len(cate_scores) == 0:
            return None
        # category labels.
        inds = inds.nonzero()
        cate_labels = inds[:, 1]

        size_trans = cate_labels.new_tensor(seg_num_grids).pow(2).cumsum(0)
        strides = cate_scores.new_ones(size_trans[-1])
        n_stage = len(seg_num_grids)
        strides[:size_trans[0]] *= strides_solo[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= strides_solo[ind_]
        strides = strides[inds[:, 0]]

        # masks.
        seg_preds = seg_preds[inds[:, 0]]
        seg_masks = seg_preds > test_cfg['mask_thr']
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # maskness.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > test_cfg['nms_pre']:
            sort_inds = sort_inds[:test_cfg['nms_pre']]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # Matrix NMS
        cate_scores = self.matrix_nms(seg_masks, cate_labels, cate_scores,
                                      kernel=test_cfg['kernel'], sigma=test_cfg['sigma'], sum_masks=sum_masks)

        # filter.
        keep = cate_scores >= test_cfg['update_thr']
        if keep.sum() == 0:
            return None
        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > test_cfg['max_per_img']:
            sort_inds = sort_inds[:test_cfg['max_per_img']]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]

        seg_masks = F.interpolate(seg_preds,
                                  size=ori_shape[:2],
                                  mode='bilinear').squeeze(0)
        seg_masks = seg_masks > test_cfg['mask_thr']
        return seg_masks, cate_labels, cate_scores


    def __get_masks(self, result, num_classes=80):
        import pycocotools.mask as mask_util
        for cur_result in result:
            masks = [[] for _ in range(num_classes)]
            if cur_result is None:
                return masks
            seg_pred = cur_result[0].cpu().numpy().astype(np.uint8)
            cate_label = cur_result[1].cpu().numpy().astype(np.int)
            cate_score = cur_result[2].cpu().numpy().astype(np.float)
            num_ins = seg_pred.shape[0]
            for idx in range(num_ins):
                cur_mask = seg_pred[idx, ...]
                rle = mask_util.encode(
                    np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
                rst = (rle, cate_score[idx])
                masks[cate_label[idx]].append(rst)
            return masks


    def create_dataset(self):
        self.anno = COCO('{}/annotations/instances_val2017.json'.format(self.options.get_data_path()))
        return read_dataset(self.anno.getImgIds())


    def load_data(self, img_id):
        img_info = self.anno.loadImgs([img_id])[0]
        img_name = '{}/val2017/{}'.format(self.options.get_data_path(), img_info['file_name'])
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
        return SOLOItem(img, metas, img_id)


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

        input_name = sess.get_inputs()[0].name
        output_names = []
        for node in sess.get_outputs():
            output_names.append(node.name)
        cate_preds, seg_preds, featmap = sess.run(output_names, {input_name: datas})

        assert len(items) == 1
        item = items[0]
        item.cate_preds = cate_preds
        item.seg_preds = seg_preds
        item.featmap = featmap
        return item

        # TODO : drop
        # print("---datas.shape:",datas.shape)
        # print("---cate_preds:",cate_preds.shape)
        # print("---seg_preds:",seg_preds.shape)
        # print("---featmap:",featmap.shape)
        # ---datas.shape: (1, 3, 800, 1216)
        # ---cate_preds: (3872, 80)
        # ---seg_preds: (3872, 200, 304)
        # ---featmap: (1, 1600, 200, 304)
        # cate_preds = np.expand_dims(cate_preds, axis=0)
        # seg_preds = np.expand_dims(seg_preds, axis=0)
        # featmap = np.expand_dims(featmap, axis=0)
        # return [cate_preds,seg_preds,featmap]


    def postprocess(self, item):
        cate_preds, seg_preds, featmap = torch.from_numpy(item.cate_preds),torch.from_numpy(item.seg_preds),torch.from_numpy(item.featmap)

        # TODO : drop
        # print("cate_preds:",cate_preds.shape)
        # print("seg_preds:",seg_preds.shape)
        # print("featmap:",featmap.shape)
        # cate_preds: (3872, 80)
        # seg_preds: (3872, 200, 304)
        # featmap: (1, 1600, 200, 304)

        # nms
        seg_single = self.__get_seg_single(
            cate_preds, seg_preds, featmap_size=featmap.size()[-2:], img_shape=item.metas['img_shape'], ori_shape=tuple(item.metas['ori_shape']))
        seg = self.__get_masks([seg_single], num_classes=self.options.get_num_classes())

        # get item segmentation result and score
        result = []
        for label in range(len(seg)):
            masks = seg[label]
            for i in range(len(masks)):
                mask_score = masks[i][1]
                segm = masks[i][0]
                data = dict()
                data['image_id'] = item.img_id
                data['score'] = float(mask_score)
                data['category_id'] = self.anno.getCatIds()[label]
                segm['counts'] = segm['counts'].decode()
                data['segmentation'] = segm
                result.append(data)
        return result


    def eval(self, collections):
        results = sum(collections,[])
        coco_dets = self.anno.loadRes(results)
        cocoEval = COCOeval(self.anno, coco_dets, 'segm')
        cocoEval.params.imgIds = self.anno.getImgIds()
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        return {}


