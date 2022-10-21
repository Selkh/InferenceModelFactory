#!/usr/bin/python3
#
# Copyright 2021 Enflame. All Rights Reserved.
#

from shutil import rmtree, move
from os import mkdir
from os.path import dirname, exists, join
from absl import flags, app
from zipfile import ZipFile

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_path', default = None, help = 'where the directory containing the download files')
  flags.DEFINE_string('output_path', default = 'data', help = 'where to extract the dataset to')

def main(unused_argv):
  assert exists(join(FLAGS.input_path, 'coco2017.zip'))
  if exists(FLAGS.output_path): rmtree(FLAGS.output_path)
  zip_file = ZipFile(join(FLAGS.input_path, 'coco2017.zip'), 'r')
  zip_file.extractall(dirname(FLAGS.output_path))
  move(join(dirname(FLAGS.output_path), 'coco2017'), FLAGS.output_path)

if __name__ == "__main__":
  add_options()
  app.run(main)

