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


import common.utils as _  # suppose to import common.util before models
import onnx_models
from common.model_factory import ModelFactory, ModelUnimplementException
from common.options import new_options


def concat_string(frame: str, model: str):
    return frame + '-' + model


def create_model_by_argument():
    options = new_options()
    options.add_argument('--frame', required=True)
    options.add_argument('--model', required=True)

    args = options.parse_known_args()[0]
    name = concat_string(args.frame, args.model)

    all_models = ModelFactory.display_all()
    if name in all_models:
        return ModelFactory.get(name).new_model()
    else:
        raise ModelUnimplementException()


def create_model_by_name(name: str):
    all_models = ModelFactory.display_all()
    if name in all_models:
        return ModelFactory.get(name).new_model()
    else:
        raise ModelUnimplementException()
