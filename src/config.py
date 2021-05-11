from os.path import join
import os


# Hyperparameters
MAX_LEN = 280
TRAIN_BATCH_SIZE = 6
VALID_BATCH_SIZE = 2
EPOCHS = 8

# PATHS
BASE_PATH = os.getcwd()
BASE_DATA_PATH = join(BASE_PATH, "Data")
MODELS_PATHS = join(BASE_PATH, "models")

# MODELS
FINBERT_UNCASED = join(MODELS_PATHS, "finbert-uncased")
MORTBERT_UNCASED = join(MODELS_PATHS, "mortbert-uncased")
BERT_UNCASED_PATH = join(MODELS_PATHS, "bert-base-uncased")

# Vocabularies
FINBERT_UNCASED_VOCAB = join(FINBERT_UNCASED, "vocab.txt")
MORTBERT_UNCASED_VOCAB = join(MORTBERT_UNCASED, "vocab.txt")
BERT_UNCASED_VOCAB = join(BERT_UNCASED_PATH, "vocab.txt")

# DATA PATHS
TRAINING_FILE = join(BASE_DATA_PATH, "NER_DF", "ner_dataset.csv")
ACC_FILE = join(BASE_DATA_PATH, "NER_data")
CHECKPOINTS_META_PATH = join(BASE_DATA_PATH, "Checkpoints", "std_data.bin")
CHECKPOINTS_MODEL_PATH = join(BASE_DATA_PATH, "Checkpoints", "model.bin")
CHECKPOINTS_LOG = join(BASE_DATA_PATH, "Checkpoints", "test.log")
