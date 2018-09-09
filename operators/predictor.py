#  coding=utf-8
import os
from scipy.io.wavfile import write
from tqdm import tqdm
from base.base_predict import BasePredict
from utils.signal_process import spectrogram2wav


class TacotronPredictor(BasePredict):
    def __init__(self, sess, model, data_loader, config, logger):
        super(TacotronPredictor, self).__init__(sess, model, data_loader, config, logger)

    def predict_epoch(self):
        """
        implement the logic of epoch:
        -loop on the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        loop = tqdm(range(self.config.iter_per_epoch))
        for _ in loop:
            self.predict_step()

    def predict_step(self):
        names, mags = self.sess.run([self.model.name, self.model.mag_hat])
        for i, mag in enumerate(mags):
            name = names[i].decode('utf-8', errors='ignore')
            print('File {}.wav is being generated ...'.format(name))
            audio = spectrogram2wav(mag)
            write(os.path.join(self.config.sample_dir, '{}.wav'.format(name)), self.config.sample_rate, audio)
