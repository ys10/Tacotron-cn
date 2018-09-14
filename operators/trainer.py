#  coding=utf-8
from tqdm import tqdm
from base.base_train import BaseTrain
from utils.signal_process import plot_alignment


class TacotronTrainer(BaseTrain):
    def __init__(self, sess, model, data_loader, config, logger):
        super(TacotronTrainer, self).__init__(sess, model, data_loader, config, logger)

    def train_epoch(self):
        """
        Implement the logic of epoch:
        -loop on the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        loop = tqdm(range(self.config.iter_per_epoch))
        for _ in loop:
            loss, mel_loss, mag_loss, lr, mel_gt, mel_hat, mag_gt, mag_hat = self.train_step()
            cur_it = self.model.global_step_tensor.eval(self.sess)
            self.logger.summarize(cur_it, scope=self.config.mode, summaries_dict={'loss': loss,
                                                                                  'mel_loss': mel_loss,
                                                                                  'mag_loss': mag_loss,
                                                                                  'lr': lr,
                                                                                  })
            self.logger.summarize_static_img(scope=self.config.mode, summaries_dict={'mel_gt': mel_gt,
                                                                                     'mel_hat': mel_hat,
                                                                                     'mag_gt': mag_gt,
                                                                                     'mag_hat': mag_hat})

            # plot attention alignments
            if cur_it % 100 == 0:
                # plot the first alignments for logging
                al = self.sess.run(self.model.alignments)
                plot_alignment(al[0], cur_it, self.config.align_dir)
        self.model.save(self.sess)

    def train_step(self):
        """
        Implement the logic of the train step
        - run the tf.Session
        - return any metrics you need to summarize
        """
        _, loss, mel_loss, mag_loss, lr, mel_gt, mel_hat, mag_gt, mag_hat = self.sess.run([self.model.train_op,
                                                                                           self.model.loss,
                                                                                           self.model.mel_loss,
                                                                                           self.model.mag_loss,
                                                                                           self.model.lr,
                                                                                           self.model.mel_gt,
                                                                                           self.model.mel_hat,
                                                                                           self.model.mag_gt,
                                                                                           self.model.mag_hat,
                                                                                           ])
        return loss, mel_loss, mag_loss, lr, mel_gt, mel_hat, mag_gt, mag_hat
