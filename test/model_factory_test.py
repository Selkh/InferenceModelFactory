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

import sys
sys.path.append("..")
from common.model_factory import *

class FakeModelFactory(ModelFactory):
    name = "fake-fmk"
    model = "undefined"

class Fake1Model(FakeModelFactory):
    model = "fake1"

models = FakeModelFactory.display_all()

assert len(models) == 1
assert FakeModelFactory.is_registered("fake-fmk-fake1")
c = FakeModelFactory.get("fake-fmk-fake1")
assert c is Fake1Model

FakeModelFactory.reset()

assert len(models) == 0
