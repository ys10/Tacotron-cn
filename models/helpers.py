import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import Helper


def _go_frames(batch_size, output_dim):
    """
    Returns all-zero <GO> frames for a given batch size and output dimension
    :param batch_size:
    :param output_dim:
    :return:
    """
    return tf.tile([[0.0]], [batch_size, output_dim])


# Adapted from tf.contrib.seq2seq.GreedyEmbeddingHelper
class TacotronInferenceHelper(Helper):
    def __init__(self, batch_size, output_dim, reduction):
        with tf.name_scope('TacotronTestHelper'):
            self._batch_size = batch_size
            self._output_dim = output_dim
            self._end_token = tf.tile([0.0], [output_dim * reduction])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        return tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim)

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """
        Stop on EOS. Otherwise, pass the last output as the next input and pass through state.
        :param time:
        :param outputs:
        :param state:
        :param sample_ids:
        :param name:
        :return:
        """
        with tf.name_scope('TacotronInferenceHelper'):
            finished = tf.reduce_all(tf.equal(outputs, self._end_token), axis=1)
            # Feed last output frame as next input. outputs is [N, output_dim * r]
            next_inputs = outputs[:, -self._output_dim:]
            return finished, next_inputs, state


class TacotronTrainingHelper(Helper):

    def __init__(self, batch_size, output_dim, reduction, targets):
        with tf.name_scope('TacotronTrainingHelper'):
            self._batch_size = batch_size
            self._output_dim = output_dim

            # Feed every r-th target frame as input
            self._targets = targets[:, reduction - 1::reduction, :]

            # Use full length for every target because we don't want to mask the padding frames
            num_steps = tf.shape(self._targets)[1]
            self._lengths = tf.tile([num_steps], [self._batch_size])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return np.int32

    def initialize(self, name=None):
        return tf.tile([False], [self._batch_size]), _go_frames(self._batch_size, self._output_dim)

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope(name or 'TacoTrainingHelper'):
            finished = (time + 1 >= self._lengths)
            next_inputs = self._targets[:, time, :]
            return finished, next_inputs, state
