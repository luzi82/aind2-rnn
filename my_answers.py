import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # calculate the output length
    output_len = len(series)-window_size
    
    # containers for input/output pairs
    X = []
    y = []
    
    # create the X, Y array
    X = [ series[i:i+window_size] for i in range(output_len) ]
    y = [ series[i+window_size] for i in range(output_len) ]
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    model = Sequential()
    # layer 1 uses an LSTM module with 5 hidden units
    model.add(LSTM(5, input_shape=(window_size, 1)))
    # layer 2 uses a fully connected module with one unit
    model.add(Dense(1))
    
    return model


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text
    
    # create a set of char in text
    char_set = set([c for c in text])
    
    # display all char exist in text
    char_set=sorted(char_set)
    print('all characters: {}'.format(''.join(char_set)))
    
    # create white list of characters to keep
    punctuation_char_set = set([' ', '!', ',', '.', ':', ';', '?'])
    englist_char_set = set([ c for c in string.ascii_lowercase])
    whitelist_char_set = punctuation_char_set | englist_char_set
    
    # remove as many non-english characters and character sequences as you can 
    
    # convert all non-white-list-ed characters to space
    text = ''.join([ c if c in whitelist_char_set else ' ' for c in text ])
    
    # shorten any extra dead space created above
    while True:
        text_len = len(text)
        text = text.replace('  ',' ')
        if text_len == len(text):
            break
    
    return text


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # calculate the output length
    output_len = len(text)-window_size
    
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # create the X, Y array
    inputs  = [ text[i:i+window_size] for i in range(output_len) if i % step_size == 0 ]
    outputs = [ text[i+window_size]   for i in range(output_len) if i % step_size == 0 ]
        
    return inputs,outputs
