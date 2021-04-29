from NER_model import NER
from src.Preprocessing import *
import logging

logging.basicConfig(filename='test.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

if __name__ == '__main__':
    # Preprocess part
    Preprocessor = NER_preprocessing(lines_sent=1, spliter="lines")
    Preprocessor.create_csv_NER()
    logging.info('Preprocessing Complete')

    #NER part
    print('NER Started....')
    model = NER(encoding="utf-8", base_model="bert-base-uncased",
                pos_dropout=0.3, tag_dropout=0.3,
                pos_dropout_2=0.3, tag_dropout_2=0.3,
                architecture="complex", middle_layer=100)
    model.training()