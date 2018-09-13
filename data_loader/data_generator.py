# coding=utf-8
import os
import codecs
import re
import numpy as np
from utils.signal_process import load_spectrogram
from utils.embd_load import EmbdMapper, OrigEmbdMapper


def _list2str(temp_list):
    return ''.join(temp_list)


def _text_normalize(text, vocab, unk_char):
    # replace any oov(out-of-vision) char with a special char: unk_char.
    vocab = _list2str(vocab)
    text = re.sub('[^{}]'.format(vocab), unk_char, text)
    # replace continuous spaces with only one space.
    text = re.sub('[ ]+', ' ', text)
    return text


class TrainDataGenerator(object):
    def __init__(self, config):
        self.config = config
        self.embd_mapper = EmbdMapper(self.config)
        self.char2idx = self.embd_mapper.get_char2idx()
        self.lookup_table = self.embd_mapper.get_lookup_table()
        self.vocab = self.embd_mapper.get_vocab()
        # load data here
        if config.language == 'cn':  # Chinese
            self.fpaths, self.text_lengths, self.texts, self.sample_count = self._load_cn_data(self.config)
        else:  # English
            self.fpaths, self.text_lengths, self.texts, self.sample_count = self._load_en_data(self.config)

    def next(self):
        for idx in range(self.sample_count):
            # load wav
            fpath = self.fpaths[idx]
            fname, mel, mag = load_spectrogram(fpath, self.config.reduction)
            mel_length = len(mel)
            mel = mel.tostring()
            mag = mag.tostring()
            # load text
            text = self.texts[idx]
            text_length = self.text_lengths[idx]
            yield {'text': text, 'text_length': text_length,
                   'mel': mel, 'mel_length': mel_length,
                   'mag': mag}

    def _load_cn_data(self, config):
        # Parse
        fpaths, text_lengths, texts = [], [], []
        transcript = os.path.join(config.data_path, config.transcription_file)
        lines = codecs.open(transcript, 'r', 'utf-8').readlines()
        sent_count = len(lines)

        for line in lines:
            fname, text, *_ = line.strip().split('|')
            # wave file path
            fpath = os.path.join(config.data_path, fname)
            fpaths.append(fpath)
            # text
            text = _text_normalize(text, self.vocab, self.config.unk_char)
            text = [self.char2idx[char] for char in text]
            text_lengths.append(len(text))
            texts.append(np.array(text, np.int32).tostring())
        return fpaths, text_lengths, texts, sent_count

    def _load_en_data(self, config):
        # Parse
        fpaths, text_lengths, texts = [], [], []
        transcript = os.path.join(config.data_path, config.transcription_file)
        lines = codecs.open(transcript, 'r', 'utf-8').readlines()
        sent_count = len(lines)

        for line in lines:
            fname, _, text = line.strip().split('|')
            # wave file path
            fpath = os.path.join(config.data_path, 'wavs', fname + '.wav')
            fpaths.append(fpath)
            # text
            text = _text_normalize(text, self.vocab, self.config.unk_char) + 'E'  # E: EOS
            text = [self.char2idx[char] for char in text]
            text_lengths.append(len(text))
            texts.append(np.array(text, np.int32).tostring())
        return fpaths, text_lengths, texts, sent_count


class PredictDataGenerator(object):
    def __init__(self, config):
        self.config = config
        self.embd_mapper = EmbdMapper(self.config)
        self.char2idx = self.embd_mapper.get_char2idx()
        self.lookup_table = self.embd_mapper.get_lookup_table()
        self.vocab = self.embd_mapper.get_vocab()
        # load data here
        if self.config == 'cn':  # Chinese
            self.names, self.text_lengths, self.texts, self.sample_count = self._load_cn_data(self.config)
        else:  # English
            self.names, self.text_lengths, self.texts, self.sample_count = self._load_en_data(self.config)

    def next(self):
        for idx in range(self.sample_count):
            # load text
            name = self.names[idx]
            text = self.texts[idx]
            text_length = self.text_lengths[idx]
            yield {'name': name, 'text': text, 'text_length': text_length}

    def _load_cn_data(self, config):
        # Parse
        names, text_lengths, texts = [], [], []
        test_file_path = os.path.join(config.data_path, config.test_file)
        lines = codecs.open(test_file_path, 'r', 'utf-8').readlines()[1:]
        sent_count = len(lines)
        print('sent_count_{}'.format(sent_count))

        for line in lines:
            name, text = line.strip().split('|')
            print('text{}'.format(text))
            # name
            names.append(name)
            # text
            text = _text_normalize(text, self.vocab, self.config.unk_char)
            text = [self.char2idx[char] for char in text]
            print('idx_{}'.format(text))
            # text length
            text_lengths.append(len(text))
            texts.append(np.array(text, np.int32).tostring())
        return names, text_lengths, texts, sent_count

    def _load_en_data(self, config):
        # Parse
        names, text_lengths, texts = [], [], []
        test_file_path = os.path.join(config.data_path, config.test_file)
        lines = codecs.open(test_file_path, 'r', 'utf-8').readlines()[1:]
        sent_count = len(lines)
        print('sent_count_{}'.format(sent_count))

        for line in lines:
            name, text = line.strip().split('|')
            print('text{}'.format(text))
            # name
            names.append(name)
            # text
            text = _text_normalize(text, self.vocab, self.config.unk_char) + 'E'  # E: EOS
            text = [self.char2idx[char] for char in text]
            print('idx_{}'.format(text))
            # text length
            text_lengths.append(len(text))
            texts.append(np.array(text, np.int32).tostring())
        return names, text_lengths, texts, sent_count


class OrigPredictDataGenerator(object):
    def __init__(self, config):
        self.config = config
        self.embd_mapper = OrigEmbdMapper(self.config)
        self.char2idx = self.embd_mapper.get_char2idx()
        self.lookup_table = self.embd_mapper.get_lookup_table()
        self.vocab = self.embd_mapper.get_vocab()
        # load data here
        if self.config == 'cn':  # Chinese
            self.names, self.text_lengths, self.texts, self.sample_count = self._load_cn_data(self.config)
        else:  # English
            self.names, self.text_lengths, self.texts, self.sample_count = self._load_en_data(self.config)

    def next(self):
        for idx in range(self.sample_count):
            # load text
            name = self.names[idx]
            text = self.texts[idx]
            text_length = self.text_lengths[idx]
            yield {'name': name, 'text': text, 'text_length': text_length}

    def _load_cn_data(self, config):
        # Parse
        names, text_lengths, texts = [], [], []
        test_file_path = os.path.join(config.data_path, config.test_file)
        lines = codecs.open(test_file_path, 'r', 'utf-8').readlines()[1:]
        sent_count = len(lines)
        print('sent_count_{}'.format(sent_count))

        for line in lines:
            name, text = line.strip().split('|')
            print('text{}'.format(text))
            # name
            names.append(name)
            # text
            text = _text_normalize(text, self.vocab, self.config.unk_char)
            text = [self.char2idx[char] for char in text]
            print('idx_{}'.format(text))
            # text length
            text_lengths.append(len(text))
            texts.append(text)
        return names, text_lengths, texts, sent_count

    def _load_en_data(self, config):
        # Parse
        names, text_lengths, texts = [], [], []
        test_file_path = os.path.join(config.data_path, config.test_file)
        lines = codecs.open(test_file_path, 'r', 'utf-8').readlines()[1:]
        sent_count = len(lines)
        print('sent_count_{}'.format(sent_count))

        for line in lines:
            name, text = line.strip().split('|')
            print('text{}'.format(text))
            # name
            names.append(name)
            # text
            text = _text_normalize(text, self.vocab, self.config.unk_char) + 'E'  # E: EOS
            text = [self.char2idx[char] for char in text]
            print('idx_{}'.format(text))
            # text length
            text_lengths.append(len(text))
            texts.append(text)
        return names, text_lengths, texts, sent_count
