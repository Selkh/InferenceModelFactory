# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import onnxruntime as ort
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.data.audio_to_text import AudioToCharDataset
from nemo.collections.asr.metrics.wer import WER
from collections import OrderedDict
import json
from typing import Union, Iterator, List

