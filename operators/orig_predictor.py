#  coding=utf-8
import os
import tensorflow as tf
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm
from base.base_predict import BasePredict
from utils.signal_process import spectrogram2wav


def embedding_lookup(lookup_table, text_idices):
    return [list(map(lambda x: lookup_table[x], text_idices))]


class OrigTacotronPredictor(BasePredict):
    def __init__(self, sess, model, config, logger, data_generator):
        super(OrigTacotronPredictor, self).__init__(sess, model, None, config, logger)
        self.lookup_table = data_generator.lookup_table
        self.data_generator = data_generator.next()

    def predict_epoch(self):
        """
        implement the logic of epoch:
        -loop on the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        loop = range(self.config.iter_per_epoch)
        for data_dict in self.data_generator:
            # data_dict = self.data_generator.next()  # {'name': name, 'text': text, 'text_length': text_length}
            name = data_dict['name']
            text = embedding_lookup(self.lookup_table, data_dict['text'])
            self.predict_step(name, text)

    def predict_step(self, name, text):
        ## mel
        mel_hat = np.zeros([1, 200, self.config.n_mels * self.config.reduction], np.float32)  # hp.n_mels*hp.r
        for j in tqdm(range(200)):
            _mel_hat = self.sess.run(self.model.mel_hat, feed_dict={self.model.text: text, self.model.mel: mel_hat})
            mel_hat[:, j, :] = _mel_hat[:, j, :]

        mags = self.sess.run(self.model.mag_hat, feed_dict={self.model.mel_hat: mel_hat})
        mag = mags[0]
        print('File {}.wav is being generated ...'.format(name))
        audio = spectrogram2wav(mag)
        write(os.path.join(self.config.sample_dir, '{}.wav'.format(name)), self.config.sample_rate, audio)
        print('File {}.wav has been saved.'.format(name))
