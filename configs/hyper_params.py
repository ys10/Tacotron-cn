# coding=utf-8


class EmbdConfig:
    # embedding
    embd_path = 'data/embd/sgns.renmin.char.reduce'
    unk_char = ' '  # replace unknown char with space.


class SignalProcessConfig:
    # signal processing
    sample_rate = 22050  # Sample rate.
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sample_rate * frame_shift)  # samples.
    win_length = int(sample_rate * frame_length)  # samples.
    n_mels = 80  # Number of Mel banks to generate
    power = 1.2  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    pre_emphasis = .97  # or None
    max_db = 100
    ref_db = 20


class ModelConfig:
    # model
    embd_size = 300
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highway_net_blocks = 4
    reduction = 3  # Reduction factor. Paper => 2, 3, 5
    dropout_rate = .5


class TrainConfig(EmbdConfig, SignalProcessConfig, ModelConfig):
    """
    Training hyper parameters
    """
    # data
    data_path = 'data/chinese-single-speaker-speech-dataset'
    transcription_file = 'transcript.txt'

    # training scheme
    mode = 'orig_train'
    is_training = True
    epochs = 100
    lr = 0.001  # Initial learning rate.
    checkpoint_dir = 'logdir/orig_checkpoints/'
    align_dir = 'logdir/alignments'
    summary_dir = 'logdir/orig_summary'
    batch_size = 32
    max_to_keep = 50


class PredictConfig(EmbdConfig, SignalProcessConfig, ModelConfig):
    """
    Predict hyper parameters
    """
    # data
    data_path = 'data/chinese-single-speaker-speech-dataset'
    test_file = 'test.txt'
    max_duration = 10.0

    # training scheme
    mode = 'predict'
    is_training = False
    lr = 0.001  # Initial learning rate.
    checkpoint_dir = 'logdir/orig_checkpoints/'
    align_dir = 'logdir/alignments'
    sample_dir = 'logdir/samples'
    summary_dir = 'logdir/orig_summary'
    batch_size = 5
    max_to_keep = 50


class OrigPredictConfig(EmbdConfig, SignalProcessConfig, ModelConfig):
    """
    Predict hyper parameters
    """
    # data
    data_path = 'data/chinese-single-speaker-speech-dataset'
    test_file = 'test.txt'
    max_duration = 10.0

    # training scheme
    mode = 'orig_predict'
    is_training = False
    lr = 0.001  # Initial learning rate.
    checkpoint_dir = 'logdir/orig_checkpoints/'
    align_dir = 'logdir/alignments'
    sample_dir = 'logdir/samples'
    summary_dir = 'logdir/orig_summary'
    batch_size = 1
    max_to_keep = 50
