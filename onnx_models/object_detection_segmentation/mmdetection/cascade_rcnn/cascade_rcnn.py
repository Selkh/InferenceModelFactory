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
            

class CascadeRcnnMMDFactory(OnnxModelFactory):
    model = "cascade_rcnn-mmd"

    def new_model():
        return CascadeRcnnMMD()




class CascadeRcnnMMD(MMdetectionModel):
    def __init__(self):
        super(CascadeRcnnMMD, self).__init__()
        self.options = self.get_options()
        self.options.add_argument('--model_path',
                                  default='./model/cascade_rcnn-resnet50_fpn_pytorch_20e-mmdetection-op13-fp32-N-topk_static.onnx',
                                  help='onnx path')
    
    
        