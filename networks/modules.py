# coding=utf-8
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import GRUCell, MultiRNNCell, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper, dynamic_decode
from networks.rnn_wrappers import ConcatOutputAndAttentionWrapper, OutputProjectionWrapper
from models.helpers import TacotronInferenceHelper


def learning_rate_decay(init_lr, global_step, warm_up_steps=4000.):
    """
    Noam scheme from tensor2tensor
    :param init_lr: original learning rate
    :param global_step: global step
    :param warm_up_steps: warm up step
    :return:
    """
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warm_up_steps ** 0.5 * tf.minimum(step * warm_up_steps ** -1.5, step ** -0.5)


def embed(inputs, lookup_table, scope='embedding', reuse=None):
    """
    Embeds a given tensor.
    :param inputs: A `Tensor` with type `int32` or `int64` containing the ids to be looked up in `lookup table`.
    :param lookup_table: A single tensor representing the complete embedding tensor.
    :param scope: Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return: A `Tensor` with one more rank than inputs's. The last dimensionality should be `num_units`.
    """
    with tf.variable_scope(scope, reuse=reuse):
        return tf.nn.embedding_lookup(lookup_table, inputs)


def bn(inputs,
       is_training=True,
       activation_fn=None,
       scope='bn',
       reuse=None):
    """
    Applies batch normalization.
    :param inputs:
        A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
        If type is `bn`, the normalization is over all but the last dimension. Or if type is `ln`,
         the normalization is over the last dimension.
        Note that this is different from the native `tf.layers.batch_normalization`.
        For this I recommend you change a line in ``tensorflow/layers/python/layers/layer.py` as follows.
        Before: mean, variance = nn.moments(inputs, axis, keep_dims=True)
        After: mean, variance = nn.moments(inputs, [-1], keep_dims=True)
    :param is_training: Whether or not the layer is in training mode.
    :param activation_fn: Activation function.
    :param scope: Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return: A tensor with the same shape and data dtype as `inputs`.
    """
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims

    # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
    # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
    if inputs_rank in [2, 3, 4]:
        if inputs_rank == 2:
            inputs = tf.expand_dims(inputs, axis=1)
            inputs = tf.expand_dims(inputs, axis=2)
        elif inputs_rank == 3:
            inputs = tf.expand_dims(inputs, axis=1)

        outputs = tf.layers.batch_normalization(inputs=inputs,
                                                training=is_training,
                                                # scope=scope,
                                                reuse=reuse,
                                                fused=True)
        # restore original shape
        if inputs_rank == 2:
            outputs = tf.squeeze(outputs, axis=[1, 2])
        elif inputs_rank == 3:
            outputs = tf.squeeze(outputs, axis=1)
    else:  # fallback to naive batch norm
        outputs = tf.layers.batch_normalization(inputs=inputs,
                                                is_training=is_training,
                                                scope=scope,
                                                reuse=reuse,
                                                fused=False)
    if activation_fn is not None:
        outputs = activation_fn(outputs)

    return outputs


def conv1d(inputs,
           filters=None,
           size=1,
           rate=1,
           padding='same',
           use_bias=False,
           activation_fn=None,
           scope='conv1d',
           reuse=None):
    """

    :param inputs: A 3-D tensor with shape of [batch, time, depth].
    :param filters: An int. Number of outputs (=activation maps)
    :param size: An int. Filter size.
    :param rate: An int. Dilation rate.
    :param padding: Either `same` or `valid` or `causal` (case-insensitive).
    :param use_bias: A boolean.
    :param activation_fn:
    :param scope: Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return:
    """
    with tf.variable_scope(scope):
        if padding.lower() == 'causal':
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = 'valid'

        if filters is None:
            filters = inputs.get_shape().as_list()[-1]

        params = {'inputs': inputs, 'filters': filters, 'kernel_size': size,
                  'dilation_rate': rate, 'padding': padding, 'activation': activation_fn,
                  'use_bias': use_bias, 'reuse': reuse}

        outputs = tf.layers.conv1d(**params)
    return outputs


def conv1d_banks(inputs, embed_size, num_banks=16, is_training=True, scope='conv1d_banks', reuse=None):
    """
    Applies a series of conv1d separately.
    :param inputs: A 3d tensor with shape of [N, T, C]
    :param embed_size: An int. The dimension of embedding size.
    :param num_banks: An int. The size of conv1d banks.
        That is, the `inputs` are convolved with K filters: 1, 2, ..., K.
    :param is_training: A boolean. This is passed to an argument of `bn`.
    :param scope: Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return: A 3d tensor with shape of [N, T, K*Hp.embed_size//2].
    """
    with tf.variable_scope(scope, reuse=reuse):
        outputs = conv1d(inputs, embed_size // 2, 1)  # k=1
        for k in range(2, num_banks + 1):  # k = 2...num_banks
            with tf.variable_scope('num_{}'.format(k)):
                output = conv1d(inputs, embed_size // 2, k)
                outputs = tf.concat((outputs, output), -1)
        outputs = bn(outputs, is_training=is_training, activation_fn=tf.nn.relu)
    return outputs  # (N, T, Hp.embed_size//2*num_banks)


def gru(inputs, num_units=None, bidirection=False, scope="gru", reuse=None):
    """
    Applies a GRU.
    :param inputs: A 3d tensor with shape of [N, T, C].
    :param num_units: An int. The number of hidden units.
    :param bidirection: A boolean. If True, bidirectional results are concatenated.
    :param scope: Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return: If bidirection is True, a 3d tensor with shape of [N, T, 2*num_units], otherwise [N, T, num_units].
    """
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list()[-1]

        cell = tf.nn.rnn_cell.GRUCell(num_units)
        if bidirection:
            cell_bw = tf.nn.rnn_cell.GRUCell(num_units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, dtype=tf.float32)
            return tf.concat(outputs, 2)
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            return outputs


def orig_attention_decoder(inputs,
                           memory,
                           num_units=None,
                           n_mels=80,
                           reduction=1,
                           scope='attention_decoder',
                           reuse=None):
    """
    Applies a GRU to 'inputs', while attending 'memory'.
    :param inputs: A 3d tensor with shape of [N, T', C']. Decoder inputs.
    :param memory: A 3d tensor with shape of [N, T, C]. Outputs of encoder network.
    :param num_units: An int. Attention size.
    :param n_mels: An int. Number of Mel banks to generate.
    :param reduction: An int. Reduction factor. Paper => 2, 3, 5.
    :param scope: Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return: A 3d tensor with shape of [N, T, num_units].
    """
    with tf.variable_scope(scope, reuse=reuse):
        attention_mechanism = BahdanauAttention(num_units, memory)
        decoder_cell = tf.nn.rnn_cell.GRUCell(num_units)
        attention_cell = AttentionWrapper(decoder_cell, attention_mechanism, num_units, alignment_history=True)
        decoder_outputs, final_decoder_state = tf.nn.dynamic_rnn(attention_cell,
                                                                 inputs, dtype=tf.float32)  # ( N, T', 16)
        # Decoder RNNs
        decoder_outputs += gru(decoder_outputs, num_units, bidirection=False,
                               scope='mel_decoder_gru_1')  # (N, T_y/r, E)
        decoder_outputs += gru(decoder_outputs, num_units, bidirection=False,
                               scope='mel_decoder_gru_2')  # (N, T_y/r, E)

        # Outputs => (N, T_y/r, n_mels*r)
        decoder_outputs = tf.layers.dense(decoder_outputs, n_mels * reduction)
    return decoder_outputs, final_decoder_state


def attention_decoder(inputs,
                      memory,
                      num_units=None,
                      batch_size=1,
                      inputs_length=None,
                      n_mels=80,
                      reduction=1,
                      default_max_iters=200,
                      is_training=True,
                      scope='attention_decoder',
                      reuse=None):
    """
    Applies a GRU to 'inputs', while attending 'memory'.
    :param inputs: A 3d tensor with shape of [N, T', C']. Decoder inputs.
    :param memory: A 3d tensor with shape of [N, T, C]. Outputs of encoder network.
    :param num_units: An int. Attention size.
    :param batch_size: An int. Batch size.
    :param inputs_length: An int. Memory length.
    :param n_mels: An int. Number of Mel banks to generate.
    :param reduction: An int. Reduction factor. Paper => 2, 3, 5.
    :param default_max_iters: Default max iteration of decoding.
    :param is_training: running mode.
    :param scope: Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return: A 3d tensor with shape of [N, T, num_units].
    """
    with tf.variable_scope(scope, reuse=reuse):
        # params setting
        if is_training:
            max_iters = None
        else:
            max_iters = default_max_iters
        # max_iters = default_max_iters
        if num_units is None:
            num_units = inputs.get_shape().as_list()[-1]

        # Decoder cell
        decoder_cell = tf.nn.rnn_cell.GRUCell(num_units)

        # Attention
        # [N, T_in, attention_depth]
        attention_cell = AttentionWrapper(
            decoder_cell,
            BahdanauAttention(num_units, memory),
            alignment_history=True)

        # Concatenate attention context vector and RNN cell output into a 2*attention_depth=512D vector.
        # [N, T_in, 2*attention_depth]
        concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)

        # Decoder (layers specified bottom to top):
        # [N, T_in, decoder_depth]
        decoder_cell = MultiRNNCell([
            OutputProjectionWrapper(concat_cell, num_units),
            ResidualWrapper(GRUCell(num_units)),
            ResidualWrapper(GRUCell(num_units))],
            state_is_tuple=True)

        # Project onto r mel spectrogram (predict r outputs at each RNN step):
        output_cell = OutputProjectionWrapper(decoder_cell, n_mels * reduction)

        decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

        if is_training:
            # helper = TacotronTrainingHelper(batch_size, n_mels, reduction, inputs)
            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=inputs,
                sequence_length=inputs_length,
                time_major=False
            )
        else:
            helper = TacotronInferenceHelper(batch_size, n_mels, reduction)

        decoder = BasicDecoder(output_cell, helper, decoder_init_state)
        # [N, T_out/r, M*r]
        (decoder_outputs, _), final_decoder_state, _ = dynamic_decode(
            decoder,
            maximum_iterations=max_iters)

    return decoder_outputs, final_decoder_state


def pre_net(inputs, embed_size, dropout_rate, num_units=None, is_training=True, scope='pre_net', reuse=None):
    """
    Pre-net for Encoder and Decoder1.
    :param inputs: A 2D or 3D tensor.
    :param embed_size: An int. The dimension of embedding.
    :param dropout_rate: A float. Drop out rate, between 0 and 1.
    :param num_units: A list of two integers. or None.
    :param is_training: A python boolean.
    :param scope: Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return: A 3D tensor of shape [N, T, num_units/2].
    """
    if num_units is None:
        num_units = [embed_size, embed_size // 2]

    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.dense(inputs, units=num_units[0], activation=tf.nn.relu, name='dense_1')
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training, name='dropout_1')
        outputs = tf.layers.dense(outputs, units=num_units[1], activation=tf.nn.relu, name='dense_2')
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training, name='dropout_2')
    return outputs  # (N, ..., num_units[1])


def highway_net(inputs, num_units=None, scope='highway_net', reuse=None):
    """
    Highway networks, see https://arxiv.org/abs/1505.00387
    :param inputs: A 3D tensor of shape [N, T, W].
    :param num_units: An int or `None`.
        Specifies the number of units in the highway layer or uses the input size if `None`.
    :param scope: Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return: A 3D tensor of shape [N, T, W].
    """
    if not num_units:
        num_units = inputs.get_shape()[-1]

    with tf.variable_scope(scope, reuse=reuse):
        h = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name='dense_1')
        t = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
                            bias_initializer=tf.constant_initializer(-1.0), name='dense_2')
        outputs = h * t + inputs * (1. - t)
    return outputs


def encoder_cbhg_module(inputs, embd_size, num_banks, num_highway_net_blocks, is_training=True):
    """
    CBHG module.
    :param inputs: A 3d tensor with shape of [N, T_x, E/2], with dtype of int32. Module inputs.
    :param embd_size: An int. The dimension of embedding.
    :param num_banks: An int. The number of filter banks.
    :param num_highway_net_blocks: An int. The number of highway net blocks.
    :param is_training: Whether or not the layer is in training mode.
    :return: A collection of Hidden vectors. So-called memory. Has the shape of (N, T_x, E).
    """
    # 1D-Convolution banks
    enc = conv1d_banks(inputs, embd_size // 2, num_banks, is_training=is_training)  # (N, T_x, K*E/2)

    # Max pooling
    enc = tf.layers.max_pooling1d(enc, pool_size=2, strides=1, padding='same')  # (N, T_x, K*E/2)

    # 1D-Convolution projections
    enc = conv1d(enc, filters=embd_size // 2, size=3, scope='conv1d_1')  # (N, T_x, E/2)
    enc = bn(enc, is_training=is_training, activation_fn=tf.nn.relu, scope='conv1d_1')

    enc = conv1d(enc, filters=embd_size // 2, size=3, scope='conv1d_2')  # (N, T_x, E/2)
    enc = bn(enc, is_training=is_training, scope='conv1d_2')

    enc += inputs  # (N, T_x, E/2) # residual connections

    # Highway Networks
    for i in range(num_highway_net_blocks):
        enc = highway_net(enc, num_units=embd_size // 2,
                          scope='highway_net_{}'.format(i))  # (N, T_x, E/2)

    # Bidirectional GRU
    memory = gru(enc, num_units=embd_size // 2, bidirection=True)  # (N, T_x, E)
    return memory


def decoder_cbhg_module(inputs, embd_size, num_banks, n_mels, num_highway_net_blocks, is_training=True):
    """
    CBHG module.
    :param inputs: A 3d tensor with shape of [N, T_x, E/2], with dtype of int32. Module inputs.
    :param embd_size: An int. The dimension of embedding.
    :param num_banks: An int. The number of filter banks.
    :param n_mels: An int. Number of Mel banks to generate.
    :param num_highway_net_blocks: An int. The number of highway net blocks.
    :param is_training: Whether or not the layer is in training mode.
    :return: A collection of Hidden vectors. So-called memory. Has the shape of (N, T_x, E).
    """
    # 1D-Convolution banks
    dec = conv1d_banks(inputs, embd_size // 2, num_banks, is_training=is_training)  # (N, T_x, K*E/2)

    # Max pooling
    dec = tf.layers.max_pooling1d(dec, pool_size=2, strides=1, padding='same')  # (N, T_x, K*E/2)

    # 1D-Convolution projections
    dec = conv1d(dec, filters=embd_size // 2, size=3, scope='conv1d_1')  # (N, T_x, E/2)
    dec = bn(dec, is_training=is_training, activation_fn=tf.nn.relu, scope='conv1d_1')

    dec = conv1d(dec, filters=n_mels, size=3, scope='conv1d_2')  # (N, T_x, E/2)
    dec = bn(dec, is_training=is_training, scope='conv1d_2')

    # Extra affine transformation for dimensionality sync
    dec = tf.layers.dense(dec, embd_size // 2)  # (N, T_y, E/2)

    # Highway Networks
    for i in range(num_highway_net_blocks):
        dec = highway_net(dec, num_units=embd_size // 2,
                          scope='highway_net_{}'.format(i))  # (N, T_x, E/2)

    # Bidirectional GRU
    dec = gru(dec, num_units=embd_size // 2, bidirection=True)  # (N, T_x, E)
    return dec


def encoder(inputs,
            embd_size,
            dropout_rate,
            num_banks,
            num_highway_net_blocks,
            is_training=True,
            scope='encoder',
            reuse=None):
    """
    :param inputs: A 3d tensor with shape of [N, T_x, E], with dtype of int32. Encoder inputs.
    :param embd_size: An int. The dimension of embedding.
    :param dropout_rate: A float. Drop out rate, between 0 and 1.
    :param num_banks: An int. The number of filter banks.
    :param num_highway_net_blocks: An int. The number of highway net blocks.
    :param is_training: Whether or not the layer is in training mode.
    :param scope: Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return: A collection of Hidden vectors. So-called memory. Has the shape of (N, T_x, E).
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Encoder pre-net
        pre_net_out = pre_net(inputs, embd_size, dropout_rate, is_training=is_training)  # (N, T_x, E/2)

        # Encoder CBHG module
        encoder_outputs = encoder_cbhg_module(pre_net_out, embd_size, num_banks, num_highway_net_blocks, is_training)

    return encoder_outputs


def orig_mel_decoder(inputs,
                     memory,
                     num_unit,
                     dropout_rate,
                     n_mels,
                     reduction,
                     is_training=True,
                     scope='mel_decoder',
                     reuse=None):
    """
    :param inputs: A 3d tensor with shape of [N, T_y/r, n_mels(*r)]. Shifted log mel-spectrogram of sound files.
    :param memory: A 3d tensor with shape of [N, T_x, E].
    :param num_unit: An int. The dimension of RNN unit.
    :param dropout_rate: A float. Drop out rate, between 0 and 1.
    :param n_mels: An int. Number of Mel banks to generate.
    :param reduction: An int. Reduction factor. Paper => 2, 3, 5.
    :param is_training: Whether or not the layer is in training mode.
    :param scope: Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return: Predicted log mel-spectrogram tensor with shape of [N, T_y/r, n_mels*r].
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        inputs = pre_net(inputs, num_unit, dropout_rate, is_training=is_training)  # (N, T_y/r, E/2)

        # Attention RNN
        decoder_outputs, state = orig_attention_decoder(inputs,
                                                        memory,
                                                        num_units=num_unit,
                                                        n_mels=n_mels,
                                                        reduction=reduction,
                                                        )  # (N, T_y/r, num_unit)

        mel_hats = decoder_outputs
        # Reshape mel
        # mel_hats = tf.reshape(mel_hats, [batch_size, -1, n_mels])

        # # for attention monitoring
        alignments = tf.transpose(state.alignment_history.stack(), [1, 2, 0])

    return mel_hats, alignments


def mel_decoder(inputs,
                memory,
                num_unit,
                dropout_rate,
                n_mels,
                reduction,
                batch_size,
                inputs_length=None,
                is_training=True,
                scope='mel_decoder',
                reuse=None):
    """
    :param inputs: A 3d tensor with shape of [N, T_y/r, n_mels(*r)]. Shifted log mel-spectrogram of sound files.
    :param memory: A 3d tensor with shape of [N, T_x, E].
    :param inputs_length: An int. Memory length.
    :param num_unit: An int. The dimension of RNN unit.
    :param dropout_rate: A float. Drop out rate, between 0 and 1.
    :param n_mels: An int. Number of Mel banks to generate.
    :param reduction: An int. Reduction factor. Paper => 2, 3, 5.
    :param batch_size: An int. Batch size.
    :param is_training: Whether or not the layer is in training mode.
    :param scope: Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return: Predicted log mel-spectrogram tensor with shape of [N, T_y/r, n_mels*r].
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        inputs = pre_net(inputs, num_unit, dropout_rate, is_training=is_training)  # (N, T_y/r, E/2)

        # Attention RNN
        decoder_outputs, state = attention_decoder(inputs,
                                                   memory,
                                                   num_units=num_unit,
                                                   batch_size=batch_size,
                                                   inputs_length=inputs_length,
                                                   n_mels=n_mels,
                                                   reduction=reduction,
                                                   is_training=is_training
                                                   )  # (N, T_y/r, num_unit)
        mel_hats = decoder_outputs
        # Reshape mel
        # mel_hats = tf.reshape(mel_hats, [batch_size, -1, n_mels])

        # # for attention monitoring
        alignments = tf.transpose(state[0].alignment_history.stack(), [1, 2, 0])

    return mel_hats, alignments


def mag_decoder(inputs,
                n_mels,
                embd_size,
                num_banks,
                num_highway_net_blocks,
                n_fft,
                is_training=True,
                scope='mag_decoder',
                reuse=None):
    """
    Decoder Post-processing net = CBHG
    :param inputs:
        A 3d tensor with shape of [N, T_y/r, n_mels*r].
        Log magnitude spectrogram of sound files.
        It is recovered to its original shape.
    :param n_mels: An int. Number of Mel banks to generate.
    :param embd_size: An int. The dimension of embedding.
    :param num_banks: An int. The number of filter banks.
    :param num_highway_net_blocks: An int. The number of highway net blocks.
    :param n_fft: An int. Number of fft points (samples).
    :param is_training: Whether or not the layer is in training mode.
    :param scope: Optional scope for `variable_scope`.
    :param reuse: Boolean, whether to reuse the weights of a previous layer by the same name.
    :return: Predicted linear spectrogram tensor with shape of [N, T_y, 1+n_fft//2].
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Restore shape -> (N, Ty, n_mels)
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, n_mels])
        # Decoder post-processing net CBHG module
        dec = decoder_cbhg_module(inputs, embd_size, num_banks, n_mels, num_highway_net_blocks, is_training)

        # Outputs => (N, T_y, 1+n_fft//2)
        outputs = tf.layers.dense(dec, 1 + n_fft // 2)

    return outputs
