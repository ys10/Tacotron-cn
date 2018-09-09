# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl


class ConcatOutputAndAttentionWrapper(rnn_cell_impl.RNNCell):
    """
    Concatenates RNN cell output with the attention context vector.

    This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
    attention_layer_size=None and output_attention=False. Such a cell's state will include an
    'attention' field that is the context vector.
    """

    def __init__(self, cell):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention

    def compute_output_shape(self, input_shape):
        # TODO
        return None

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        return tf.concat([output, res_state.attention], axis=-1), res_state

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


class OutputProjectionWrapper(rnn_cell_impl.RNNCell):
    """
    Operator adding an output projection to the given cell.

    Note: in many cases it may be more efficient to not use this wrapper,
    but instead concatenate the whole sequence of your outputs in time,
    do the projection on this batch-concatenated sequence, then split it
    if needed or directly feed into a softmax.
    """

    def __init__(self, cell, output_size, activation=None, reuse=None):
        """Create a cell with output projection.

        Args:
          cell: an RNNCell, a projection to output_size is added to it.
          output_size: integer, the size of the output after projection.
          activation: (optional) an optional activation function.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.

        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if output_size is not positive.
        """
        super(OutputProjectionWrapper, self).__init__(_reuse=reuse)
        rnn_cell_impl.assert_like_rnncell("cell", cell)
        if output_size < 1:
            raise ValueError("Parameter output_size must be > 0: %d." % output_size)
        self._cell = cell
        self._output_size = output_size
        self._activation = activation
        self._linear = None

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._output_size

    def compute_output_shape(self, input_shape):
        # TODO
        return None

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def call(self, inputs, state):
        """Run the cell and output projection on inputs, starting from state."""
        output, res_state = self._cell(inputs, state)
        projected = tf.layers.dense(output, self.output_size, activation=self._activation)
        return projected, res_state
