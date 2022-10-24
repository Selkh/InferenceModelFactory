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
#!/usr/bin/python
# -*- coding: utf-8 -*-

from onnx_models.classification.classification import ClassificationModel
from onnx_models.base import OnnxModelFactory


class Res2net5026w4sPPFactory(OnnxModelFactory):
    model = "res2net50_26w_4s-pp"

    def new_model():
        return Res2net5026w4sPP()


class Res2net5026w4sPP(ClassificationModel):
    def __init__(self):
        super(Res2net5026w4sPP, self).__init__()
