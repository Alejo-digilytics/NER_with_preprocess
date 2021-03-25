## NER with BERT like models

This repository fine tunes for a NER task. There are two main classes: NER_preprocessing which preprocess 
text files to get a data frame, and NER which trains a BER-like model using such a data frame. 
This class also produces accuracy and loss pots and contains a prediction method for testing


### Launch
To use this repository you must verify the requirements listed in requirements.txt
This can be done moving to the working directory and running the following command on terminal 
`pip install -r requirements.txt`

#### Pytorch and cuda
One of the libraries used here is pytorch.
The version depends on the computer and must be compatible with the cuda installed in the computer as well as the OS.
Pay attention to the fact that the current Pytorch version do not support cuda 11.1 even it exits already.
At most you can use cuda 11.0, which can be found here:
`https://developer.nvidia.com/cuda-11.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal`

It can be install using the following command:
`pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html`

If you want to check your cuda you can do it as follows:

    1. Check cuda for windows: run the following command in the cmd "nvcc --version"
    2. Check cuda for Linux or Mac: assuming that cat is your editor run "cat /usr/local/cuda/version.txt",
    or the version.txt localization if other

Downloading pytorch: go to `https://pytorch.org/get-started/locally/` and follow the instructions for the download.

#### BERT-like models
The BERT-like model must be added in the folder models containing the following files:

    1. config.json: the model's configuration
    2. pytorch_model.bin: weights of the model 3.special_tokens.json: Not mandatory, since it is created alongthe process if there is a vocab.txt file
    3. vocab.txt: vocabulary of the model in a column with rows number the id of the model


### NER Dataset

The NER data set must be added manually into the folder `Data/NER_data` as txt files.
This data is preprocessed using the class NER_preprocessing from `src/Preprocessing.py`.

There are two ways of splitting the data. The first one based on the sentences and a special one
for this data set considering text and tables separately.

Output:

    Option Simple: Data Frame with columns "Sentences", "Words", "POS" and "Tag"
        "Sentences": number of the sentence in the word and empty the rest of the sentence
        "Words":token of the word
        "POS": Part of speech ffor the token (based on Spacy)
        "Tag": BILUO tag with entity and position

    Option Simple: Data Frame with columns "Sentences", "Words", "POS", "Tag" and "NER"
        "Sentences": number of the sentence in the word and empty the rest of the sentence
        "Words":token of the word
        "POS": Part of speech ffor the token (based on Spacy)
        "Tag": BILUO tag with entity and position
        "NER": general entities for the token given by Spacy

This data frame is saved in `Data/NER_DF` where it will be then used to fine tune the model.