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


# class Factory(object):
#     map = {}
# 
#     def __new__(cls, *args, **kwargs):
#         import pdb;pdb.set_trace()
#         r = object.__new__(cls, *args, **kwargs)
#         cls_name = cls.__name__
#         print(cls_name)
# 
#         def register(cls):
#             map[cls_name] = cls
#             return cls
# 
#         return register(cls)
# 
#     @classmethod
#     def all(self):
#         return self.map
# 
# class Base(Factory):
#     pass
# 
# class Test(Base):
#     __name__ = 'test'
# 
# t = Test()
# map = Factory.all()
# print(map)


from factory import create_model_by_argument, create_model_by_name
from monitor import Monitor

# engine = create_model_by_argument()
# monitor = Monitor()
# monitor.Execute(engine)
# 
# engine1 = create_model_by_argument()
# monitor.Execute(engine1)

engine = create_model_by_name('onnx-resnet50')
engine1 = create_model_by_name('onnx-resnet50')
#engine0 = create_model_by_name('onnx-yolo')
