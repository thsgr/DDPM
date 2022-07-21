# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compute FAD between two multivariate Gaussian."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os 

from .fad_utils import *

flags.DEFINE_string("background_stats", None,
                    "Tf record containing the background stats (mu sigma).")
flags.DEFINE_string("test_stats", None,
                    "Tf record containing the test stats (mu sigma).")

flags.mark_flags_as_required(["background_stats", "test_stats"])

FLAGS = flags.FLAGS


def main(argv):
  del argv
  mu_bg, sigma_bg = read_mean_and_covariances(FLAGS.background_stats)
  mu_test, sigma_test = read_mean_and_covariances(FLAGS.test_stats)
  fad = frechet_distance(mu_bg, sigma_bg, mu_test, sigma_test)
  print("FAD: %f" % fad)

  result_file = FLAGS.test_stats.replace('stats', 'results')
  result_dir = result_file.rsplit('.')[0].rsplit('/')[:-1]
  result_dir = '/'.join(p for p in result_dir)

  # Make sure results folder exists
  if not Path(result_dir).exists():
    os.mkdir(result_dir)
    print(f"Created {result_dir} folder \n")

  os.system(f'echo "fad : {fad}" > {result_file}_fad.txt')


if __name__ == "__main__":
  app.run(main)
