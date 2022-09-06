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

# suppose to import common.util first
import common.utils as utils
from factory import create_model_by_argument, create_model_by_name
from monitor import Monitor

model = create_model_by_argument()
monitor = Monitor()

model1 = create_model_by_name('onnx-resnet50')

monitor.Execute(model)
monitor.Execute(model1)
