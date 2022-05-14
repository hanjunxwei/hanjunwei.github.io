import tensorflow as tf
import numpy as np
from functools import reduce


def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the
    sentences to build the train and test data separately. Create a vocabulary
    dictionary that maps all the unique tokens from your train and test data as
    keys to a unique integer value. Then vectorize your train and test data based
    on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id
    form), test (1-d list or array with testing words in vectorized/id form),
    vocabulary (dict containing word->index mapping)
    """
    
    # Loading data from file path
    with open(train_file) as f:
        train_lines = f.readlines()
    with open(test_file) as f:
        test_lines = f.readlines()
        
    # Concatenate and Split
    ## Train
    train = ""
    for i in train_lines:
        train+= " " + i
    
    train = train.split()
    
    
    ## Test
    test = ""
    for i in test_lines:
        test+= " " + i
    
    test = test.split()
    
    ## remove punctuation
    ## remove punctuation
    #punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    #for i in punc:
    #    if i in test:
    #        test = list(filter(lambda x: x != i, test))
    #
    #for i in punc:
    #    if i in train:
    #        train = list(filter(lambda x: x != i, train))    
    
    
    # Create dictionary
    dictionary = {}
    ## Extract unique word
    unique_words = list(set(test+train))
    ## create dictionary
    for i in range(len(unique_words)):
        dictionary[unique_words[i]] = i
        
    # Tokenize
    ## train
    tokenize_train = []
    for i in train:
        tokenize_train.append(dictionary[i])
    ## test
    tokenize_test = []
    for i in test:
        tokenize_test.append(dictionary[i])
    # TODO: return training tokens, testing tokens, and the vocab dictionary.
    return np.array(tokenize_train),np.array(tokenize_test),dictionary