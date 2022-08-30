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

__all__ = ['get_default_options', 'new_options', 'reset_options', 'remove_options']

import sys
import threading
import argparse

class Options:
    """Agent mode, an Options instance handle an ArgumentParser, which could be add_argument by users in '__init__' of Model instance. Some methods of Parser is wrapped so that user could call familiar interface to achieve same function"""
    def __init__(self):
        self.__parser = argparse.ArgumentParser(allow_abbrev=False)

    def get_parser(self):
        # Usually where's no need to get parser but just in case
        return self.__parser

    def add_argument(self, *args, **kwargs):
        self.__parser.add_argument(*args, **kwargs)

    def parse_args(self, args=None, namespace=None):
        return self.__parser.parse_known_args(args, namespace)[0]

    def parse_known_args(self):
        return self.__parser.parse_known_args()

    def get(self, key):
        if key.startswith('__'):
            k = key
        else:
            k = '__' + key

        if hasattr(self, k):
            return getattr(self, k)
        else:
            return None

class OptionsStack(threading.local):
    """Use a stack to manage Options objects"""
    def __init__(self):
        self.options_stack = []

    def get_default_options(self):
        if self.options_stack:
            return self.options_stack[-1]
        else:
            options = Options()
            self.options_stack.append(options)
            return options

    def reset(self):
        self.options_stack = []

    def push(self, options: Options):
        self.options_stack.append(options)

    def remove(self, options: Options):
        self.options_stack.remove(options)
        del options

_options_stack = OptionsStack()


def get_default_options():
    return _options_stack.get_default_options()

def new_options():
    options = Options()
    _options_stack.push(options)
    return options

def reset_options():
    _options_stack.reset()

def remove_options(options: Options):
    _options_stack.remove(options)


#def create_set_function(*args):
#    def set_attr(self, value):
#        setattr(self, key, value)
#
#    return set_attr
#
#def create_get_function(*args):
#    def get_attr(self, key):
#        return getattr(self, key)
#
#    return get_attr
#
#
#def make_options():
#
#    args = sys.argv[1:]
#    nargs = len(args)
#
#    if nargs % 2:
#        raise ValueError('')
#
#    for i in range(0, nargs, 2):
#        if args[i].startswith('--'):
#            key = args[i][2:]
#            if key.startswith('-'):
#                raise ValueError('Valid key name: {}'.format(key))
#
#            value = args[i + 1]
#            if value.startswith('-'):
#                raise ValueError('Valid value: {}'.format(value))
#
#            set_func_name = 'set_' + key
#            new_set_func = create_set_function(key)
#            setattr(Option, set_func_name, new_set_func)
#
#            get_func_name = 'get_' + key
#            new_get_func = create_get_function()
#            setattr(Option, get_func_name, new_get_func)
#
#            setattr(Option, '__' + key, value)
#
#    opts = Options()
#    return opts
