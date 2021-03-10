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

"""Exposes the ogbg-molpcba dataset in a convenient format."""

from typing import Dict, NamedTuple
import jraph
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class GraphsTupleSize(NamedTuple):
  """Helper class to represent padding and graph sizes."""
  n_node: int
  n_edge: int
  n_graph: int


def get_datasets(batch_size: int) -> Dict[str, tf.data.Dataset]:
  """Returns datasets of batched GraphsTuples, organized by split."""
  if batch_size <= 0:
    raise ValueError('Batch size must be > 0.')

  # Obtain the original datasets.
  datasets = get_raw_datasets()

  # Process each split separately.
  for split in datasets:

    # Convert to GraphsTuple.
    datasets[split] = datasets[split].map(
        convert_to_graphs_tuple,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True)

  # Compute the padding budget for the requested batch size.
  budget = estimate_padding_budget_for_batch_size(datasets['train'], batch_size)

  # Process each split separately.
  for split, dataset_split in datasets.items():

    # Repeat and shuffle the training split.
    # We cache the validation and test splits, since those are small.
    if split == 'train':
      dataset_split = dataset_split.repeat()
      dataset_split = dataset_split.shuffle(100)
      drop_remainder = True
      cache = False
    else:
      drop_remainder = False
      cache = True

    # Batch.
    dataset_split = batch_and_pad_dataset(
        dataset_split, budget, drop_remainder=drop_remainder)

    # Cache datasets.
    if cache:
      dataset_split = dataset_split.cache()

    # Pre-fetch batches.
    dataset_split = dataset_split.prefetch(tf.data.AUTOTUNE)
    datasets[split] = dataset_split
  return datasets


def get_raw_datasets() -> Dict[str, tf.data.Dataset]:
  """Returns datasets as tf.data.Dataset, organized by split."""
  ds_builder = tfds.builder('ogbg_molpcba')
  ds_builder.download_and_prepare()
  datasets = ds_builder.as_dataset()
  return datasets


def estimate_padding_budget_for_batch_size(
    dataset: tf.data.Dataset,
    batch_size: int,
    num_estimation_graphs: int = 100) -> GraphsTupleSize:
  """Estimates the padding budget for a dataset of unbatched GraphsTuples.

  The padding budget can be passed directly to `batch_and_pad_dataset()`
  to perform the actual batching.

  Args:
    dataset: A dataset of unbatched GraphsTuples.
    batch_size: The intended batch size. Note that no batching is performed by
      this function.
    num_estimation_graphs: How many graphs to take from the dataset to estimate
      the distribution of number of nodes and edges per graph.

  Returns:
    padding_budget: The padding budget for batching and padding the graphs
    in this dataset to the given batch size.
  """

  def next_multiple_of_64(val: int):
    """Returns the next multiple of 64 after val."""
    return 64 * (1 + int(val // 64))

  if batch_size <= 0:
    raise ValueError('Batch size must be > 0.')

  num_nodes = []
  num_edges = []
  for graph in dataset.take(num_estimation_graphs).as_numpy_iterator():
    graph_size = get_graphs_tuple_size(graph)
    if graph_size.n_graph != 1:
      raise ValueError('Dataset contains batched GraphTuples.')

    num_nodes.append(graph_size.n_node)
    num_edges.append(graph_size.n_edge)

  num_nodes_per_graph_estimate = np.mean(num_nodes)
  num_edges_per_graph_estimate = np.mean(num_edges)

  padding_budget = GraphsTupleSize(
      n_node=next_multiple_of_64(num_nodes_per_graph_estimate * batch_size) - 1,
      n_edge=next_multiple_of_64(num_edges_per_graph_estimate * batch_size),
      n_graph=batch_size)
  return padding_budget


def convert_to_graphs_tuple(graph: Dict[str, tf.Tensor]) -> jraph.GraphsTuple:
  """Converts a dictionary of tf.Tensors to a GraphsTuple."""
  labels = graph['labels']
  return jraph.GraphsTuple(
      n_node=graph['num_nodes'],
      n_edge=graph['num_edges'],
      nodes=graph['node_feat'],
      edges=graph['edge_feat'],
      senders=graph['edge_index'][:, 0],
      receivers=graph['edge_index'][:, 1],
      globals=tf.expand_dims(labels, axis=0),
  )


def specs_from_graphs_tuple(graph: jraph.GraphsTuple):
  """Returns a tf.TensorSpec corresponding to this graph."""

  def get_tensor_spec(array: np.ndarray):
    shape = list(array.shape)
    dtype = array.dtype
    return tf.TensorSpec(shape=shape, dtype=dtype)

  specs = {}
  for field in [
      'nodes', 'edges', 'senders', 'receivers', 'globals', 'n_node', 'n_edge'
  ]:
    field_sample = getattr(graph, field)
    specs[field] = get_tensor_spec(field_sample)
  return jraph.GraphsTuple(**specs)


def get_graphs_tuple_size(graph: jraph.GraphsTuple):
  """Returns the number of nodes, edges and graphs in a GraphsTuple."""
  return GraphsTupleSize(
      n_node=np.sum(graph.n_node),
      n_edge=np.sum(graph.n_edge),
      n_graph=np.shape(graph.n_node)[0])


def batch_and_pad_dataset(dataset: tf.data.Dataset,
                          budget: GraphsTupleSize,
                          drop_remainder: bool = True) -> tf.data.Dataset:
  """Batches and pads a dataset of unbatched GraphsTuples.

  The output batched GraphsTuples will have exactly one more node
  and one more graph than the specified budget.

  Args:
    dataset: A tf.data.Dataset of single unbatched GraphsTuples.
    budget: The budget per batch of GraphsTuples.
    drop_remainder: Whether to drop the last incomplete batch.

  Returns:
    padded_dataset: A tf.data.Dataset of batched and padded GraphsTuples.
  """
  # Add one node and one graph to ensure we can always pad correctly.
  padded_shape = GraphsTupleSize(
      n_node=budget.n_node + 1,
      n_edge=budget.n_edge,
      n_graph=budget.n_graph + 1)

  # Pad an example graph to see what the output shapes will be.
  example_graph = next(dataset.as_numpy_iterator())
  example_padded_graph = jraph.pad_with_graphs(example_graph, *padded_shape)
  padded_spec = specs_from_graphs_tuple(example_padded_graph)

  def batch_and_pad_dataset_generator():
    # Initialize budget.
    remaining_budget = budget
    curr_batch = []

    # For each graph in the dataset, see if it fits into the current batch.
    # If not, yield the current batch and start a new one.
    for graph in dataset.as_numpy_iterator():
      graph_size = get_graphs_tuple_size(graph)

      # Batch is full, so we should wrap it up.
      if (remaining_budget.n_node < graph_size.n_node or
          remaining_budget.n_edge < graph_size.n_edge or
          remaining_budget.n_graph == 0):
        # Pad.
        curr_batch_as_graphs_tuple = jraph.batch(curr_batch)
        yield jraph.pad_with_graphs(curr_batch_as_graphs_tuple, *padded_shape)

        # Reset budget.
        remaining_budget = budget
        curr_batch = []

      # Add to current batch.
      curr_batch.append(graph)

      # Update statistics.
      remaining_budget = GraphsTupleSize(
          n_node=remaining_budget.n_node - graph_size.n_node,
          n_edge=remaining_budget.n_edge - graph_size.n_edge,
          n_graph=remaining_budget.n_graph - 1)

    # The remaining graphs.
    if not drop_remainder or remaining_budget.n_graph == 0:
      curr_batch_as_graphs_tuple = jraph.batch(curr_batch)
      yield jraph.pad_with_graphs(curr_batch_as_graphs_tuple, *padded_shape)

  return tf.data.Dataset.from_generator(
      batch_and_pad_dataset_generator, output_signature=padded_spec)


def get_dummy_datasets(dataset_length: int,
                       batch: bool = True) -> Dict[str, tf.data.Dataset]:
  """Returns a dummy set of datasets, useful for testing."""

  # The dummy graph.
  num_nodes = 3
  num_edges = 4
  num_graphs = 1
  node_feature_dim = 5
  edge_feature_dim = 1
  global_feature_dim = 128
  dummy_graph = jraph.GraphsTuple(
      n_node=tf.expand_dims(num_nodes, 0),
      n_edge=tf.expand_dims(num_edges, 0),
      senders=tf.zeros(num_edges, dtype=tf.int32),
      receivers=tf.ones(num_edges, dtype=tf.int32),
      nodes=tf.zeros((num_nodes, node_feature_dim)),
      edges=tf.ones((num_edges, edge_feature_dim)),
      globals=tf.ones((num_graphs, global_feature_dim), dtype=tf.int64),
  )
  graphs_spec = specs_from_graphs_tuple(dummy_graph)

  # Yields a set of graphs for the current split.
  def get_dummy_graphs():
    for _ in range(dataset_length):
      yield dummy_graph

  datasets = {}
  for split in ['train', 'validation', 'test']:
    datasets[split] = tf.data.Dataset.from_generator(
        get_dummy_graphs, output_signature=graphs_spec)

  if batch:
    batch_size = dataset_length // 2
    budget = estimate_padding_budget_for_batch_size(
        datasets['train'], batch_size, num_estimation_graphs=1)

    for split, dataset in datasets.items():
      datasets[split] = batch_and_pad_dataset(
          dataset, budget, drop_remainder=False)
  return datasets
