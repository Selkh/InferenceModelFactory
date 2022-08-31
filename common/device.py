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

class DeviceNotParseException(Exception):
    def __init__(self):
        print("Please call 'parse' before get when use class Device")

class Device:
    __slots__ = ['_name', '_type', '_id', '_cluster_ids']

    __supported_device_list = ['cpu', 'gpu', 'gcu']

    @property
    def name(self):
        if not hasattr(self, '_name'):
            raise DeviceNotParseException()
        return self._name

    @property
    def type(self):
        if not hasattr(self, '_type'):
            raise DeviceNotParseException()
        return self._type

    @property
    def id(self):
        if not hasattr(self, '_id'):
            raise DeviceNotParseException()
        return self._id

    @property
    def cluster_ids(self):
        if not hasattr(self, '_cluster_ids'):
            raise DeviceNotParseException()
        return self._cluster_ids

    @staticmethod
    def search_colon(substr: str):
        assert type(substr) is str
        return substr.count(':')

    @staticmethod
    def parse_list(substr: str):
        assert type(substr) is str
        return [int(i) for i in substr.split(',')]

    @classmethod
    def parse(cls, name: str):
        """
        For general devices, a standard device string format is '<device_type>:<id>', example as 'cpu:0', 'gpu:1'. Shorten formats is also supported, such as 'cpu', 'gpu', where device id will be set 0 as default.

        As gcu exposes 'cluster' in the interface, its format will be 'gcu:<id>:<cluster_ids>', example as 'gcu:0:0', 'gcu:0:0,1'
        where
          `id` is device id,
          `cluster_ids` is a comma-separated list of integers, such as '0', '0,1'. It represents single grain resource to be used
        Shorten format is also supported. If no specific assigned cluster ids, the whole resource will be used to compute and if no assigned device id, device 0 will be used.
        For example,
          'gcu' means the complete card 0
          'gcu:1' means the complete card 1
          'gcu::0,1' means cluster 0 and 1 on device 0
        """

        if type(cls._name) is str:
            print("Re-assign device name")

        cls._name = name

        assert type(cls._name) is str

        ncolon = cls.search_colon(cls._name)

        if ncolon > 2:
            raise ValueError('Specific device name: {} is not in supported format.'.format(name))

        if ncolon == 0: # 'cpu', 'gpu', 'gcu'
            device_type = cls._name
            if device_type not in cls.__supported_device_list:
                raise ValueError('Specific device name: {} is not supported.'.format(name))

            cls._type = device_type
            cls._id = 0
            cls._cluster_ids = [-1]
            return

        seperator_0 = cls._name.find(':')
        device_type = cls._name[:seperator_0]
        if device_type not in cls.__supported_device_list:
            raise ValueError('Specific device name: {} is not supported.'.format(name))
        cls._type = device_type

        if ncolon == 2:
            # gcu standard format
            assert device_type == 'gcu', 'Only gcu support format with double colons.'

            seperator_1 = cls._name[seperator_0 + 1:].find(':')

            if seperator_1 == 0:
                #'gcu::0,1,2'
                cls._id = 0
            else:
                cls._id = int(cls._name[seperator_0 + 1:][:seperator_1])

            cluster_ids = cls._name[seperator_0 + seperator_1 + 2:]
            cls._cluster_ids = cls.parse_list(cluster_ids)
            return

        if ncolon == 1:
            cls._id = int(cls._name[seperator_0 + 1:])
            cls._cluster_ids = [-1]
            return 

    @classmethod
    def rename(cls):
        return str(cls.type()) + ':' + str(cls.id())
