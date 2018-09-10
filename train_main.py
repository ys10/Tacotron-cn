# coding=utf-8
import tensorflow as tf
from models.tacotron_model import Tacotron
from data_loader.data_set_loader import TrainDataSetLoader
from data_loader.data_generator import TrainDataGenerator
from configs.hyper_params import TrainConfig
from utils.logger import Logger
from operators.trainer import TacotronTrainer


def train():
    train_config = TrainConfig()
    g = tf.Graph()
    with g.as_default():
        # load data
        train_data_gen = TrainDataGenerator(train_config)
        data_loader = TrainDataSetLoader(train_config, {'train': train_data_gen}, default_set_name='train')
        next_data = data_loader.next_data
        # config training hyper parameters
        train_config.iter_per_epoch = int(round(train_data_gen.sample_count / train_config.batch_size))
        # create an instance of the model you want
        model = Tacotron(train_config, next_data)
        # session setting
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            # create tensor-board logger
            logger = Logger(sess, train_config)
            # create trainer and pass all the previous components to it
            trainer = TacotronTrainer(sess, model, data_loader, train_config, logger)
            # load model if exists
            model.load(sess)
            # here you train your model
            trainer.train()
            # save model
            model.save(sess)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    '''train model'''
    train()
    tf.logging.info("Congratulations!")


if __name__ == "__main__":
    main()
