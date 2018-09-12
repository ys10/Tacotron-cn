# coding=utf-8
import tensorflow as tf


def get_data_set_from_generator(generator_func, lookup_table, padded_shapes,
                                output_types, new_shapes, epochs=1, batch_size=16, mode='train'):
    def parse_train_func(data):
        # process text
        text = data['text']
        text = tf.decode_raw(text, tf.int32)
        text = tf.nn.embedding_lookup(lookup_table, text)
        text = tf.reshape(text, new_shapes['text'])
        # process text length
        text_length = data['text_length']
        text_length = tf.cast(text_length, tf.int32)
        text_length = tf.reshape(text_length, new_shapes['text_length'])
        # process mel
        mel = data['mel']
        mel = tf.decode_raw(mel, tf.int32)
        mel = tf.reshape(mel, new_shapes['mel'])
        # process mel length
        mel_length = data['mel_length']
        mel_length = tf.cast(mel_length, tf.int32)
        mel_length = tf.reshape(mel_length, new_shapes['mel_length'])
        # process mag
        mag = data['mag']
        mag = tf.decode_raw(mag, tf.int32)
        mag = tf.reshape(mag, new_shapes['mag'])
        return {'text': text, 'text_length': text_length,
                'mel': mel, 'mel_length': mel_length,
                'mag': mag}

    def parse_predict_func(data):
        # process name
        name = data['name']
        # process text
        text = data['text']
        text = tf.decode_raw(text, tf.int32)
        text = tf.nn.embedding_lookup(lookup_table, text)
        text = tf.reshape(text, new_shapes['text'])
        # process text length
        text_length = data['text_length']
        text_length = tf.cast(text_length, tf.int32)
        text_length = tf.reshape(text_length, new_shapes['text_length'])
        return {'name': name, 'text': text, 'text_length': text_length}

    data_set = tf.data.Dataset.from_generator(generator_func,
                                              output_types=output_types)
    if mode in ['train', 'orig_train']:  # train & orig_train
        data_set = data_set.map(parse_train_func)
    else:  # predict & orig_predict
        data_set = data_set.map(parse_predict_func)
    data_set = data_set.repeat(epochs)
    data_set = data_set.padded_batch(batch_size, padded_shapes=padded_shapes)
    return data_set


class TrainDataSetLoader(object):
    def __init__(self, config, generators, default_set_name='train'):
        self.config = config
        self.generators = generators
        self.data_sets = dict()
        self.data_set_init_ops = dict()
        with tf.variable_scope('data'):
            for k in self.generators.keys():
                self.data_sets[k] = get_data_set_from_generator(
                    self.generators[k].next,
                    self.generators[k].lookup_table,
                    padded_shapes={'text': [None, self.config.embd_size],
                                   'text_length': [],
                                   'mel': [None, self.config.n_mels * self.config.reduction],
                                   'mel_length': [],
                                   'mag': [None, 1 + self.config.n_fft // 2]},
                    output_types={'text': tf.string,  # text
                                  'text_length': tf.int32,
                                  'mel': tf.string,  # mel
                                  'mel_length': tf.int32,
                                  'mag': tf.string},  # mag
                    new_shapes={'text': [-1, self.config.embd_size],
                                'text_length': [],
                                'mel': [-1, self.config.n_mels * self.config.reduction],
                                'mel_length': [],
                                'mag': [-1, 1 + self.config.n_fft // 2]},
                    epochs=self.config.epochs,
                    batch_size=self.config.batch_size,
                    mode=self.config.mode)
            self.iterator = self.data_sets[default_set_name].make_one_shot_iterator()
            self.next_data = self.iterator.get_next()
            for k in self.data_sets.keys():
                self.data_set_init_ops[k] = self.iterator.make_initializer(self.data_sets[k])


class PredictDataSetLoader(object):
    def __init__(self, config, generators, default_set_name='predict'):
        self.config = config
        self.generators = generators
        self.data_sets = dict()
        self.data_set_init_ops = dict()
        with tf.variable_scope('data'):
            for k in self.generators.keys():
                self.data_sets[k] = get_data_set_from_generator(
                    self.generators[k].next,
                    self.generators[k].lookup_table,
                    padded_shapes={'name': [],
                                   'text': [None, self.config.embd_size],
                                   'text_length': []},
                    output_types={'name': tf.string,
                                  'text': tf.string,
                                  'text_length': tf.int32},
                    new_shapes={'name': [],
                                'text': [-1, self.config.embd_size],
                                'text_length': []},
                    epochs=1,
                    batch_size=self.config.batch_size,
                    mode=self.config.mode)
            self.iterator = self.data_sets[default_set_name].make_one_shot_iterator()
            self.next_data = self.iterator.get_next()
            for k in self.data_sets.keys():
                self.data_set_init_ops[k] = self.iterator.make_initializer(self.data_sets[k])


class TempPredictDataSetLoader(object):
    def __init__(self, config, generators, default_set_name='predict'):
        self.config = config
        self.generator = generators[default_set_name]
        self.lookup_table = self.generator.lookup_table
        self.next_data = None

    def parse_text(self, text):
        # process text
        text = tf.decode_raw(text, tf.int32)
        text = tf.nn.embedding_lookup(self.lookup_table, text)
        # text = tf.reshape(text, [None, self.config.embd_size])
        return text
