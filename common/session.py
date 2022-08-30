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

from abc import ABC, abstractmethod

__all__ = ['BaseSession', 'register_session', 'SESSION_FACTORY']

class BaseSession(ABC):

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def version(self):
        pass

    @abstractmethod
    def run(self):
        pass

SESSION_FACTORY = {}

def register_session(cls):
    cls_name = cls.name(cls)
    def register_internal(cls):
        SESSION_FACTORY[cls_name] = cls
        
    return register_internal(cls)
