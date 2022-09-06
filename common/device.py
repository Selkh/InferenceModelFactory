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

class DeviceNotParseException(Exception):
    def __init__(self):
        print("Please call 'parse' before get when use class Device")


class Device:
    __slots__ = ['_name', '_type', '_id', '_cluster_ids']

    _supported_device_list = ['cpu', 'gpu', 'gcu']

    def __init__(self, name, device_type, device_id, cluster_ids):
        self._name = name
        self._type = device_type
        self._id = device_id
        self._cluster_ids = cluster_ids

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

    @staticmethod
    def parse(name: str):
        """
        For general devices, a standard device string format is
                    '<device_type>:<id>'
        As example:
                    'cpu:0', 'gpu:1'.
        Shorten formats is also supported, such as
                    'cpu', 'gpu',
            where
                device id will be set 0 as default.

        As gcu exposes 'cluster' in the interface, its format will be
                    'gcu:<id>:<cluster_ids>'
        As example:
                    'gcu:0:0', 'gcu:0:0,1'
            where
                `id`: device id,
                `cluster_ids`: a comma-separated list of integers, each
                               represents one single grain resource to be used
                               on GCU, such as '0', '0,1',
        Shorten format is also supported with some assumption:
            if no specific cluster ids, assume whole resource to be used,
            if no specific device id, assume device 0 to be used.
        As example:
          'gcu' means the complete card 0
          'gcu:1' means the complete card 1
          'gcu::0,1' means cluster 0 and 1 on device 0
        """

        # Only format string is supported for 'Device'
        assert type(name) is str

        # Number of colon used to match different pattern
        ncolon = Device.search_colon(name)

        # If more than two colons in 'name', it must be an abnormal format
        if ncolon > 2:
            raise ValueError(
                'Specific device name: {} is not in supported format. '
                'Please double check your string'.format(name))

        # Shorten format
        if ncolon == 0:  # 'cpu', 'gpu', 'gcu'
            device_type = name
            if device_type not in Device._supported_device_list:
                raise ValueError(
                    'Specific device name: {} is not supported. Supported: {}'.format(
                        name, Device._supported_device_list))

            device_id = 0
            cluster_ids = [-1]
            return Device(name=name, device_type=device_type,
                          device_id=device_id, cluster_ids=cluster_ids)

        seperator_0 = name.find(':')
        device_type = name[:seperator_0]
        if device_type not in Device._supported_device_list:
            raise ValueError(
                'Specific device name: {} is not supported.'.format(name))

        # Only GCU may contains two colons
        if ncolon == 2:
            # GCU standard format
            assert device_type == 'gcu', 'Only gcu support format with double colons.'

            seperator_1 = name[seperator_0 + 1:].find(':')

            if seperator_1 == 0:
                # 'gcu::0,1,2'
                device_id = 0
            else:
                device_id = int(name[seperator_0 + 1:][:seperator_1])

            cluster_ids = name[seperator_0 + seperator_1 + 2:]
            cluster_ids = Device.parse_list(cluster_ids)
            return Device(name=name, device_type=device_type,
                          device_id=device_id, cluster_ids=cluster_ids)

        if ncolon == 1:
            device_id = int(name[seperator_0 + 1:])
            cluster_ids = [-1]
            return Device(name=name, device_type=device_type,
                          device_id=device_id, cluster_ids=cluster_ids)

    @classmethod
    def rename(cls):
        return str(cls.type) + ':' + str(cls.id)
