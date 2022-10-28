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

from onnx_models.object_detection_segmentation.mmdetection.mmdetection import MMdetectionModel
from onnx_models.base import OnnxModelFactory
from common.model import Model
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


class SOLOMMDFactory(OnnxModelFactory):
    model = "solo-mmd"

    def new_model():
        return SOLOMMD()



class SOLOMMD(MMdetectionModel):
    def __init__(self):
        super(SOLOMMD, self).__init__()
        self.options.add_argument('--config',
                            help='test config file path')
        self.options.add_argument('--out_file',
                            default="results_solo_segm.json",
                            help='output result file')



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


    def run_internal(self, sess, items):
        datas = Model.make_batch([item.data for item in items])

        input_name = sess.get_inputs()[0].name
        output_names = []
        for node in sess.get_outputs():
            output_names.append(node.name)
        cate_preds, seg_preds, featmap = sess.run(output_names, {input_name: datas})

        assert len(items) == 1
        items[0].cate_preds = cate_preds
        items[0].seg_preds = seg_preds
        items[0].featmap = featmap

        return items

    def postprocess(self, item):
        cate_preds, seg_preds, featmap = torch.from_numpy(item.cate_preds),torch.from_numpy(item.seg_preds),torch.from_numpy(item.featmap)

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


