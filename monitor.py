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

from engine import CommonEngine

class Monitor(object):
    """builder mode"""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def Execute(self, engine: CommonEngine):
        processed_input = engine.preprocessing(self.args)
        output_data = engine.run(processed_input)
        processed_output = engine.postprocessing(output_data)
        return processed_output

    def __del__(self):
        # TODO: Generate Final Report
        pass
