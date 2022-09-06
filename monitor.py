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

import time
from common.model import Model
from common.options import Options


def create_set_function(options, key, *args, **kwargs):
    def set_attr(value):
        setattr(options, '_' + key, value)

    return set_attr


def create_get_function(options, key, *args, **kwargs):
    def get_attr():
        return getattr(options, '_' + key)

    return get_attr


class Monitor(object):
    """builder mode"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.start = time.time()
        self.count = 0

    def Execute(self, model: Model):
        self.count += 1
        print("Execute model with class name: {}".format(type(model).__name__))

        options = model.get_options()
        args = options.parse_args()
        print(args.__dict__)

        for key, value in vars(args).items():
            setattr(options, '__' + key, value)

            set_func_name = 'set_' + key
            new_set_func = create_set_function(options, key)
            setattr(options, set_func_name, new_set_func)

            get_func_name = 'get_' + key
            new_get_func = create_get_function(options, key)
            setattr(options, get_func_name, new_get_func)

        model.run()
        # processed_input = model.preprocess()
        # output_data = model.run(processed_input)
        # processed_output = model.postprocess(output_data)

        # return processed_output

    def __del__(self):
        # TODO: Generate Final Report
        end = time.time()
        print("Totally execute {} models, spend {}".format(self.count,
                                                           end - self.start))
