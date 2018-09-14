# coding=utf-8


class CNEmbdConfig:
    unk_char = ' '  # replace unknown char with space.
    # Chinese (embedding needed).
    language = 'cn'
    embd_path = 'data/embd/sgns.renmin.char.reduce'
    embd_size = 300


class ENEmbdConfig:
    unk_char = ' '  # replace unknown char with space.
    # English (one-hot is OK).
    language = 'en'
    embd_path = 'data/embd/en.one.hot'
    embd_size = 33


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
    encoder_num_banks = 16
    decoder_num_banks = 8
    num_highway_net_blocks = 4
    reduction = 3  # Reduction factor. Paper => 2, 3, 5
    dropout_rate = .5


class CNTrainConfig(CNEmbdConfig, SignalProcessConfig, ModelConfig):
    """
    Training hyper parameters
    """
    # Chinese data
    data_path = 'data/chinese-single-speaker-speech-dataset'
    transcription_file = 'transcript.txt'
    # training scheme
    mode = 'orig_train'
    is_training = True
    epochs = 100
    lr = 0.001  # Initial learning rate.
    checkpoint_dir = 'logdir/cn/orig_checkpoints/'
    align_dir = 'logdir/cn/orig_alignments'
    summary_dir = 'logdir/cn/orig_summary'
    batch_size = 32
    max_to_keep = 50


class ENTrainConfig(ENEmbdConfig, SignalProcessConfig, ModelConfig):
    """
    Training hyper parameters
    """
    # English data
    data_path = 'data/LJSpeech-1.1/'
    transcription_file = 'metadata.csv'
    # training scheme
    mode = 'orig_train'
    is_training = True
    epochs = 100
    lr = 0.001  # Initial learning rate.
    checkpoint_dir = 'logdir/en/orig_checkpoints/'
    align_dir = 'logdir/en/orig_alignments'
    summary_dir = 'logdir/en/orig_summary'
    batch_size = 32
    max_to_keep = 50


class CNPredictConfig(CNEmbdConfig, SignalProcessConfig, ModelConfig):
    """
    Predict hyper parameters
    """
    # Chinese data
    data_path = 'data/chinese-single-speaker-speech-dataset'
    test_file = 'test.txt'
    # training scheme
    mode = 'predict'
    is_training = False
    lr = 0.001  # Initial learning rate.
    checkpoint_dir = 'logdir/cn/checkpoints/'
    align_dir = 'logdir/cn/alignments'
    sample_dir = 'logdir/cn/samples'
    summary_dir = 'logdir/cn/summary'
    batch_size = 5
    max_to_keep = 50


class ENPredictConfig(ENEmbdConfig, SignalProcessConfig, ModelConfig):
    """
    Predict hyper parameters
    """
    # English data
    data_path = 'data/LJSpeech-1.1/'
    test_file = 'test.txt'
    # training scheme
    mode = 'predict'
    is_training = False
    lr = 0.001  # Initial learning rate.
    checkpoint_dir = 'logdir/en/checkpoints/'
    align_dir = 'logdir/en/alignments'
    sample_dir = 'logdir/en/samples'
    summary_dir = 'logdir/en/summary'
    batch_size = 5
    max_to_keep = 50


class OrigCNPredictConfig(CNEmbdConfig, SignalProcessConfig, ModelConfig):
    """
    Predict hyper parameters
    """
    # Chinese data
    data_path = 'data/chinese-single-speaker-speech-dataset'
    test_file = 'test.txt'
    # training scheme
    mode = 'orig_predict'
    is_training = False
    lr = 0.001  # Initial learning rate.
    checkpoint_dir = 'logdir/cn/orig_checkpoints/'
    align_dir = 'logdir/cn/orig_alignments'
    sample_dir = 'logdir/cn/orig_samples'
    summary_dir = 'logdir/cn/orig_summary'
    batch_size = 1
    max_to_keep = 50


class OrigENPredictConfig(ENEmbdConfig, SignalProcessConfig, ModelConfig):
    """
    Predict hyper parameters
    """
    # English data
    data_path = 'data/LJSpeech-1.1/'
    test_file = 'test.txt'
    # training scheme
    mode = 'orig_predict'
    is_training = False
    lr = 0.001  # Initial learning rate.
    checkpoint_dir = 'logdir/en/orig_checkpoints/'
    align_dir = 'logdir/en/orig_alignments'
    sample_dir = 'logdir/en/orig_samples'
    summary_dir = 'logdir/en/orig_summary'
    batch_size = 1
    max_to_keep = 50
