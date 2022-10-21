#!/usr/bin/python3
#
# Copyright 2022 Enflame. All Rights Reserved.
#

from shutil import rmtree
from os import mkdir
from os.path import join, exists
from absl import flags, app
from zipfile import ZipFile

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_path', default = None, help = 'where the directory containing the downloaded files is')
  flags.DEFINE_string('output_path', default = 'data', help = 'where to extract the dataset to')

def main(unused_argv):
  assert exists(join(FLAGS.input_path, 'archive.zip'))
  if exists(FLAGS.output_path): rmtree(FLAGS.output_path)
  mkdir(FLAGS.output_path)
  zip_file = ZipFile(join(FLAGS.input_path, 'archive.zip'))
  zip_file.extractall(FLAGS.output_path)

if __name__ == "__main__":
  add_options()
  app.run(main)

