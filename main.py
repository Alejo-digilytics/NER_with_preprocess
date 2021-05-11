from NER_model import NER
import logging

logging.basicConfig(filename='test.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

if __name__ == '__main__':
    """ 
    # Preprocess part
    Preprocessor = NER_preprocessing(lines_sent=3, spliter="lines")
    Preprocessor.create_csv_NER()
    print('Preprocessing Complete')
    """

    #NER part
    print('NER Started....')
    model = NER(encoding="utf-8",
                base_model="mortbert-uncased",
                tag_dropout=0.35,
                tag_dropout_2=0.35,
                architecture="complex",
                middle_layer=120)
    model.training()