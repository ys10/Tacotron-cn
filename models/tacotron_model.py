# coding=utf-8
import tensorflow as tf
from base.base_model import BaseModel
from networks.modules import encoder, mel_decoder, orig_mel_decoder, mag_decoder, learning_rate_decay


class Tacotron(BaseModel):
    def __init__(self, config, data=None):
        super(Tacotron, self).__init__(config, data)
        self.build_model()
        self.init_saver()

    def build_model(self):
        with tf.variable_scope('tacotron', reuse=tf.AUTO_REUSE):
            '''
            read data
                text: Text. (N, Tx)
                mel: Reduced mel-spectrogram. (N, Ty//r, n_mels*r)
                mag: Magnitude. (N, Ty, n_fft//2+1)
            '''
            if self.config.mode in ['train', 'orig_train']:
                self.text = tf.cast(self.data['text'], tf.float32)
                self.text_length = tf.cast(self.data['text_length'], tf.int32)
                self.mel = tf.cast(self.data['mel'], tf.float32)
                self.mel_length = tf.cast(self.data['mel_length'], tf.int32)
                self.mag = tf.cast(self.data['mag'], tf.float32)
            elif self.config.mode in ['orig_predict']:
                self.name = tf.placeholder(tf.string, shape=())
                self.text = tf.placeholder(tf.float32, shape=(None, None, self.config.embd_size))
                self.text_length = tf.placeholder(tf.int32, shape=())
                self.mel = tf.placeholder(tf.float32, shape=(None,
                                                             None,
                                                             self.config.n_mels * self.config.reduction))
                self.mel_length = None
            else:  # Synthesize 'predict'
                self.name = tf.cast(self.data['name'], tf.string)
                self.text = tf.cast(self.data['text'], tf.float32)
                self.text_length = tf.cast(self.data['text_length'], tf.int32)
                self.mel = tf.placeholder(tf.float32, shape=(None,
                                                             None,
                                                             self.config.n_mels * self.config.reduction))
                self.mel_length = None
            '''build graph'''
            # Get encoder/decoder inputs
            # (N, T_x, E)
            self.encoder_inputs = self.text
            # (N, Ty/r, n_mels*r)
            # decoder <GO> frame: tf.zeros_like(self.mel[:, :1, :])
            self.decoder_inputs = tf.concat((tf.zeros_like(self.mel[:, :1, :]), self.mel[:, :-1, :]), axis=1)
            # feed last frames only (N, Ty/r, n_mels)
            self.decoder_inputs = self.decoder_inputs[:, :, -self.config.n_mels:]

            # Networks
            with tf.variable_scope('net'):
                # Encoder
                # (N, T_x, E)
                self.encoder_outputs = encoder(self.encoder_inputs,
                                               embd_size=self.config.embd_size,
                                               dropout_rate=self.config.dropout_rate,
                                               num_banks=self.config.encoder_num_banks,
                                               num_highway_net_blocks=self.config.num_highway_net_blocks,
                                               is_training=self.config.is_training)
                # Mel decoder
                # (N, T_y//r, n_mels*r)
                if self.config.mode in ['orig_train', 'orig_predict']:
                    self.mel_hat, self.alignments = orig_mel_decoder(self.decoder_inputs,
                                                                     self.encoder_outputs,
                                                                     num_unit=self.config.embd_size,
                                                                     dropout_rate=self.config.dropout_rate,
                                                                     n_mels=self.config.n_mels,
                                                                     reduction=self.config.reduction,
                                                                     is_training=self.config.is_training)
                else:  # ['train', 'predict']
                    self.mel_hat, self.alignments = mel_decoder(self.decoder_inputs,
                                                                self.encoder_outputs,
                                                                num_unit=self.config.embd_size,
                                                                dropout_rate=self.config.dropout_rate,
                                                                n_mels=self.config.n_mels,
                                                                reduction=self.config.reduction,
                                                                batch_size=self.config.batch_size,
                                                                inputs_length=self.mel_length,
                                                                is_training=self.config.is_training)
                # Magnitude decoder: post processing mel spectrum with a CBHG module.
                # (N, T_y//r, (1+n_fft//2)*r)
                self.mag_hat = mag_decoder(self.mel_hat,
                                           n_mels=self.config.n_mels,
                                           embd_size=self.config.embd_size,
                                           num_banks=self.config.decoder_num_banks,
                                           num_highway_net_blocks=self.config.num_highway_net_blocks,
                                           n_fft=self.config.n_fft,
                                           is_training=self.config.is_training)

            '''build loss, optimizer'''
            with tf.name_scope('loss'):
                if self.config.mode in ['train', 'orig_train']:
                    # Loss
                    self.mel_loss = tf.reduce_mean(tf.abs(self.mel_hat - self.mel))
                    self.mag_loss = tf.reduce_mean(tf.abs(self.mag_hat - self.mag))
                    self.loss = self.mel_loss + self.mag_loss

                    # Training Scheme
                    self.lr = tf.Variable(self.config.lr, dtype=tf.float32)
                    # self.lr = learning_rate_decay(self.config.lr, global_step=self.global_step_tensor)
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

                    # gradient clipping
                    self.gvs = self.optimizer.compute_gradients(self.loss)
                    self.clipped = []
                    for grad, var in self.gvs:
                        grad = tf.clip_by_norm(grad, 2.)
                        self.clipped.append((grad, var))
                    self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step_tensor)
                    # self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)
            # '''build metrics'''
            with tf.name_scope('metrics'):
                if self.config.mode in ['train', 'orig_train']:
                    self.mel_gt = tf.expand_dims(self.mel, -1)
                    self.mel_hat = tf.expand_dims(self.mel_hat, -1)
                    self.mag_gt = tf.expand_dims(self.mag, -1)
                    self.mag_hat = tf.expand_dims(self.mag_hat, -1)
