#  coding=utf-8
import tensorflow as tf
import os


class Logger:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        if self.config.mode in ['train', 'orig_train']:
            self.summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, 'train'), self.sess.graph)
        else:
            self.summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, 'predict'))

    def summarize(self, step, scope='', summaries_dict=None):
        """
        Summarize scalars and images.
        :param step: the step of the summary
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        """
        with tf.variable_scope(scope):

            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                        else:
                            self.summary_placeholders[tag] = tf.placeholder('float32',
                                                                            [None] + list(value.shape[1:]), name=tag)
                        if len(value.shape) <= 1:
                            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                    summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    self.summary_writer.add_summary(summary, step)
                self.summary_writer.flush()
