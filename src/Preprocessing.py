import nltk
import pandas as pd
import os
import re
import spacy
import time
import logging
from src.tools import nlp_setup
import src.config as config

logging.basicConfig(filename='test.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')


class NER_preprocessing:
    def __init__(self, lines_sent=1, spliter="lines"):
        """
        This class preprocess data labeled on BILOU form to create a DATA FRAME for fine tuning a NER task.
        There are to possible DF options: simple, which considers Sentences, words, parts of speech and the labels
        in the text; and the combined which adds a NER column based on Spacy to complete this information for the task.
        There are two method to split the sentences: by real sentences which is based on punctuation and special
        which consider the lines with labels and those of normal text separately. Moreover this one creates the sentences
        for the tagged part based on the start and end of the sentences sin they can be split into different lines.
        Attributes:
            -  lines_sent (int: 1,2,3): the number of real sentences per sentence (for split_by_dots)
            -  spliter (str): "dots" = split_by_dots and "lines" = special_split
        Methods:
            - split_by_dots: it splits into real sentences or join of them.
            - special_split: the special split described above
            - BL_matcher: counter of starting, ending labels in text
            - label_matcher: label's extracter
            - create_csv_NER: creates the DF for NER fine tuning
            - create_csv_NER_combined
        """
        self.path_Data = os.path.join(os.getcwd(), "Data")
        self.path_NER_data = os.path.join(self.path_Data, "NER_data")
        self.path_NER_DF = os.path.join(self.path_Data, "NER_DF")
        self.NER_listed = [file for file in os.listdir(self.path_NER_data) if file.endswith(".txt")]
        self.lines_sent = lines_sent
        nlp_setup("NER")
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = self.nlp.tokenizer
        self.spliter = "lines" if spliter == "lines" else "dots"

    def split_by_dots(self, path):
        """
        This method converts text with punctuation into a list of sentences
        Input:
            - path (string): path to the txt file to split
        Output:
            - list of sentences as strings ([str,str,str, ...]): list of sentences in the txt split by punctuation
        """
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        file = open(path, "r", encoding="utf8")
        # the following is a list of strings, which are the sentences
        list_sentences = [re.sub("\s+", " ", sentence) for sentence in tokenizer.tokenize(file.read())]
        if self.lines_sent == 1:
            return list_sentences
        elif self.lines_sent == 2:
            return [list_sentences[i] + " " + list_sentences[i + 1] for i in range(0, len(list_sentences) - 1, 2)]
        elif self.lines_sent == 3:
            return [list_sentences[i] + " " + list_sentences[i + 1] + " " + list_sentences[i + 2] for i in
                    range(0, len(list_sentences) - 1, 3)]

    def special_split(self, path):
        """
        This method converts a labelled text mix of normal text and table into sentences.
        The normal text part is split based on sentences and the table part is split such that
        the start and the end of each entity are contained.
        Input:
            - path (string): path to the txt file to split
        Output:
            - list of sentences as strings ([str,str,str, ...]): list of sentences in the txt split
        """
        file = open(path, "r", encoding="utf8")

        # the following is a list of strings, which are the lines
        lines_sentences = [" ".join(sentence.split()) for sentence in file]
        if not lines_sentences:
            print("Following file was empty: ", path)
        elif len(lines_sentences) == 1:
            list_of_words = lines_sentences[0].split(" ")
            lines_sentences = []
            aux_list = []
            counter = 0
            for word in list_of_words:
                if counter % 10 == 0:
                    aux_list.append(word)
                    lines_sentences.append(" ".join(aux_list))
                    aux_list = []
                else:
                    aux_list.append(word)
                counter += 1
        else:
            pass

        current_sentence = ""
        list_sentences = []
        # lines_sentences = list(filter(lambda a: a not in ["", ",", ".", ""], lines_sentences))
        for sentence in lines_sentences:
            #  joining the new sentence to the current one, which can be just ""
            join_sentence = current_sentence + " " + sentence
            join_sentence = join_sentence.strip()

            # Case splitting current sentences and adding it.
            if len(join_sentence.split()) > (config.MAX_LEN / 2):
                list_sentences.append(current_sentence)
                current_sentence = sentence

            # Case not long enough to be added. Then we extend the sentence
            else:
                current_sentence = join_sentence

            # Last sentence case
            if sentence == lines_sentences[-1]:
                list_sentences.append(current_sentence)

        return list_sentences

    @staticmethod
    def BL_matcher(case, text):
        """ Matcher to count numbers of B, L or any labels in the text """
        if case == "B":
            return re.findall("\[\'[A-Z]{1,4}-B\'\]", text)
        elif case == "L":
            return re.findall("\[\'[A-Z]{1,4}-L\'\]", text)
        elif case == "all":
            return len(re.findall("\[\'[A-Z]{1,4}-[A-Z]\'\]", text))

    @staticmethod
    def label_matcher(word):
        """ matcher to extract the labels from the text and be available to extract semantic information """
        if word == " ":
            return "space"
        else:
            return re.search("\[\'[A-Z]{1,4}-[A-Z]\'\]", word)

    def create_csv_NER(self):
        """
        This method creates a DF with columns Sentence, Word, POS, Tag from the text files contained in the folder
        NER_data. This DF is saved in the folder NER_DF as ner_dataset.csv
        """
        logging.info(" Creating the data frame ...")
        start = time.time()
        # Initialize the df
        df = pd.DataFrame(columns=["Sentence", "Word", "POS", "Tag"])
        # Lists to fill
        l_sentence = []
        l_words = []
        l_POS = []
        l_tag = []
        counter = 0
        # loop over files
        for file in self.NER_listed:
            if self.spliter == "lines":
                list_sentences = self.special_split(os.path.join(self.path_NER_data, file))
            else:
                list_sentences = self.split_by_dots(os.path.join(self.path_NER_data, file))

            # loop over SENTENCES
            for sentence in list_sentences:
                counter_new = counter + 1
                wt = sentence.strip().split(" ")
                just_words = []
                for i in range(len(wt)):
                    # Real words case
                    if self.label_matcher(wt[i]) is None:
                        tokens = self.tokenizer(wt[i])  # Split word into tokens
                        tokens_text = [t.text for t in tokens]  # Extract the text of the tokens
                        just_words.extend(tokens_text)
                        tokens_len = len(tokens_text)  # num of tokens in the word
                        if counter == counter_new:
                            l_sentence.extend([""] * tokens_len)
                        elif tokens_len == 1:
                            l_sentence.append(counter)
                            counter = counter_new
                        else:
                            l_sentence.extend([""] * (tokens_len - 1))
                            l_sentence.insert(-tokens_len + 1, counter)
                            counter = counter_new
                        l_tag.extend(["O"] * tokens_len)
                    elif self.label_matcher(wt[i]) != "space":
                        tag = wt[i].replace("['", "").replace("']", "")
                        l_tag[-tokens_len:] = [tag] * tokens_len
                    else:
                        pass
                l_words.extend(just_words)
                sentence = " ".join(just_words).strip()
                doc = self.nlp(sentence)
                for token in doc:
                    l_POS.append(token.pos_)
        if len(l_POS) < len(l_sentence):
            padding_no = len(l_sentence) - len(l_POS)
            l_POS.extend(["O"] * padding_no)
        if len(l_POS) > len(l_sentence):
            l_POS = l_POS[:len(l_sentence)]
        df["Sentence"] = l_sentence
        df["Word"] = l_words
        df["POS"] = l_POS
        df["Tag"] = l_tag
        try:
            os.mkdir(self.path_NER_DF)
        except FileExistsError:
            time.sleep(0.001)  # Prevent high load in pathological conditions
        logging.info(" ... created")
        timer = time.time() - start
        logging.info(f" It took {timer} seconds")
        df.to_csv(path_or_buf=os.path.join(self.path_NER_DF, "ner_dataset.csv"), index=False)

    def create_csv_NER_combined(self):
        """
        This method creates a DF with columns Sentence, Word, POS, Tag, Entities from the text files contained
        in the folder NER_data. This DF is saved in the folder NER_DF as ner_dataset.csv.
        The column Entities is created based on the NER entities from Spacy
        """
        logging.info(" Creating the data frame for combined ...")
        start = time.time()
        # Init df
        df = pd.DataFrame(columns=["Sentence", "Word", "POS", "Tag", "Entities"])
        # Lists to fill
        l_sentence = []
        l_words = []
        l_POS = []
        l_tag = []
        l_NER = []
        counter = 0
        # loop over files
        for file in self.NER_listed:
            if self.spliter == "lines":
                list_sentences = self.special_split(os.path.join(self.path_NER_data, file))
            else:
                list_sentences = self.split_by_dots(os.path.join(self.path_NER_data, file))
            count_no_ner = 0
            # loop over SENTENCES
            for sentence in list_sentences:
                counter_new = counter + 1
                wt = sentence.strip().split(" ")
                just_words = []
                for i in range(len(wt)):

                    # Real words case
                    if self.label_matcher(wt[i]) is None:
                        tokens = self.tokenizer(wt[i])  # Split word into tokens, use just tokenizer, it's cheaper
                        tokens_text = [t.text for t in tokens]  # Extract the text of the tokens
                        just_words.extend(tokens_text)
                        tokens_len = len(tokens_text)  # num of tokens in the word
                        if counter == counter_new:
                            l_sentence.extend([""] * tokens_len)
                        elif tokens_len == 1:
                            l_sentence.append(counter)
                            counter = counter_new
                        else:
                            l_sentence.extend([""] * (tokens_len - 1))
                            l_sentence.insert(-tokens_len + 1, counter)
                            counter = counter_new
                        l_tag.extend(["O"] * tokens_len)
                    elif self.label_matcher(wt[i]) != "space":
                        tag = wt[i].replace("[", "").replace("]", "")
                        l_tag[-tokens_len:] = [tag] * tokens_len
                    else:
                        pass
                l_words.extend(just_words)

                # Semantic part: pos and ner already implemented in spacy
                # rejoin the sentence
                sentence = " ".join(just_words).strip()
                doc = self.nlp(sentence)

                # initialize the entities
                entities = []
                for entity in doc.ents:
                    entities.append(([t.text for t in self.tokenizer(entity.text)], entity.label_))
                entities_len = len(entities)
                if entities_len == 0:
                    count_no_ner += 1

                # We do not have always entities in the sentences so we use "" to avoid problems
                text_entity = entities[0][0] if entities_len != 0 else ""
                entity_num = 0

                # text_entity_len = 0 if text_entity = ""
                text_entity_len = len(text_entity)
                num_match = 0
                for token in doc:
                    l_POS.append(token.pos_)
                    # conditional for token in the entity
                    if token.text not in text_entity:
                        l_NER.append("O")
                        num_match = 0
                    else:
                        # we add "O" in a preventive way, in case it's a partial match
                        l_NER.append("O")
                        num_match += 1
                    # conditional to ignore the 0 case
                    if entities_len == 0:
                        pass
                    # case: las token of the entity but not last entity
                    elif num_match == text_entity_len and entity_num + 1 != entities_len:
                        del l_NER[-text_entity_len:]
                        l_NER.extend([entities[entity_num][1]] * text_entity_len)
                        entity_num += 1
                        text_entity = entities[entity_num][0]
                        text_entity_len = len(text_entity)
                        num_match = 0
                    # Case: las token of the entity and last entity
                    elif num_match == text_entity_len:
                        del l_NER[-text_entity_len:]
                        l_NER.extend([entities[entity_num][1]] * text_entity_len)
        df["Sentence"] = l_sentence
        df["Word"] = l_words
        df["POS"] = l_POS
        df["Tag"] = l_tag
        df["entities"] = l_NER
        try:
            os.mkdir(self.path_NER_DF)
        except FileExistsError:
            time.sleep(0.001)
        logging.info(" ... created")
        timer = time.time() - start
        logging.info(f" It took {timer} seconds")
        df.to_csv(path_or_buf=os.path.join(self.path_NER_DF, "ner_combined_dataset_example.csv"), index=False)
