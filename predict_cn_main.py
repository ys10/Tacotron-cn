# coding=utf-8
import tensorflow as tf
from models.tacotron_model import Tacotron
from data_loader.data_set_loader import PredictDataSetLoader
from data_loader.data_generator import PredictDataGenerator
from configs.hyper_params import CNPredictConfig
from utils.logger import Logger
from operators.predictor import TacotronPredictor


def predict():
    predict_config = CNPredictConfig()
    g = tf.Graph()
    with g.as_default():
        # load data
        predict_data_gen = PredictDataGenerator(predict_config)
        data_loader = PredictDataSetLoader(predict_config, {'predict': predict_data_gen}, default_set_name='predict')
        next_data = data_loader.next_data
        # config training hyper parameters
        predict_config.iter_per_epoch = int(round(predict_data_gen.sample_count / predict_config.batch_size))
        # create an instance of the model you want
        predict_config.lookup_table = predict_data_gen.lookup_table
        model = Tacotron(predict_config, next_data)
        # session setting
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        with tf.Session(config=session_config) as sess:
            # create tensor-board logger
            logger = Logger(sess, predict_config)
            # create predictor and pass all the previous components to it
            predictor = TacotronPredictor(sess, model, data_loader, predict_config, logger)
            # load model if exists
            model.load(sess)
            # here you use your model to predict
            predictor.predict()


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    '''predict'''
    predict()
    tf.logging.info("Congratulations!")


if __name__ == "__main__":
    main()
