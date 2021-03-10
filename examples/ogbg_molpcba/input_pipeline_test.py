# Copyright 2021 The Flax Authors.
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

"""Tests for flax.examples.ogbg_molpcba.input_pipeline."""

from absl.testing import absltest
from absl.testing import parameterized
import input_pipeline

Datasets = input_pipeline.Datasets


class InputPipelineTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dataset_length = 20
    self.datasets = input_pipeline.get_dummy_datasets(
        self.dataset_length, batch=False)

  @parameterized.parameters(
      1,
      5,
      12,
      15,
  )
  def test_estimate_padding_budget_valid(self, valid_batch_size):
    budget = input_pipeline.estimate_padding_budget_for_batch_size(
        self.datasets['train'], valid_batch_size, num_estimation_graphs=1)
    self.assertEqual(budget.n_graph, valid_batch_size)

  @parameterized.parameters(
      -1,
      0,
  )
  def test_estimate_padding_budget_invalid(self, invalid_batch_size):
    with self.assertRaises(ValueError):
      input_pipeline.estimate_padding_budget_for_batch_size(
          self.datasets['train'], invalid_batch_size, num_estimation_graphs=1)

  @parameterized.product(
      valid_batch_size=[1, 5, 12, 15],
      drop_remainder=[True, False],
  )
  def test_batch_and_pad_dataset(self, valid_batch_size: int,
                                 drop_remainder: bool):

    expected_num_batches = self.dataset_length // valid_batch_size
    remainder_size = self.dataset_length % valid_batch_size
    if not drop_remainder and remainder_size > 0:
      expected_num_batches += 1

    budget = input_pipeline.estimate_padding_budget_for_batch_size(
        self.datasets['train'], valid_batch_size, num_estimation_graphs=1)

    for dataset in self.datasets.values():
      batched_dataset = input_pipeline.batch_and_pad_dataset(
          dataset, budget, drop_remainder)

      num_batches = 0
      for batch in batched_dataset:
        # There is an extra padding graph in each batch.
        self.assertLen(batch.n_node, valid_batch_size + 1)
        self.assertLen(batch.globals, valid_batch_size + 1)
        num_batches += 1
      self.assertEqual(num_batches, expected_num_batches)


if __name__ == '__main__':
  absltest.main()
