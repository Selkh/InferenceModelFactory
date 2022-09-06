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

__all__ = ['get_default_options', 'new_options', 'reset_options',
           'remove_options']

import sys
import threading
import argparse


class Options:
    """
    Agent mode, an Options instance handle an ArgumentParser, which could be
    add_argument by users in '__init__' of Model instance. Some methods of
    'argparse.ArgumentParser' is wrapped so that users could call familiar
    interface to achieve same function
    """

    def __init__(self):
        self._parser = argparse.ArgumentParser(allow_abbrev=False)

    def get_parser(self):
        # Usually where's no need to get parser but just in case
        return self._parser

    def add_argument(self, *args, **kwargs):
        self._parser.add_argument(*args, **kwargs)

    def parse_args(self, args=None, namespace=None):
        # Todo: check whether arg starts with '_'
        return self._parser.parse_known_args(args, namespace)[0]

    def parse_known_args(self):
        return self._parser.parse_known_args()

    def get(self, key):
        if key.startswith('_'):
            k = key
        else:
            k = '_' + key

        if hasattr(self, k):
            return getattr(self, k)
        else:
            return None

    def __getitem__(self, key):
        return self.get(key)


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
