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

import argparse
import onnx
import tf
from common.model_factory import *
from common.options import get_default_options, new_options

class RepeatedCallException(Exception):
    def __init__(self):
        print("function: create_model_by_argument could only be called once in a single process. If need repeated calls, please use create_model_by_name instead")

def concat_string(frame: str, model: str):
    return frame + '-' + model


def create_model_by_argument():
    options = get_default_options()
    try:
        options.add_argument('--frame', required=True)
        options.add_argument('--model', required=True)
        options.add_argument('--device', required=False)
    #except argparse.ArgumentError as ex:
    except Exception:
        raise RepeatedCallException()

    known_args = options.parse_known_args()
    name = concat_string(known_args.frame, known_args.model)

    all = ModelFactory.display_all()
    if name in all:
        return ModelFactory.get(name).new_model()
    else:
        raise ModelUnimplementException()


def create_model_by_name(name: str):
    all = ModelFactory.display_all()
    if name in all:
        return ModelFactory.get(name).new_model()
    else:
        raise ModelUnimplementException()
