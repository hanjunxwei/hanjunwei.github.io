import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys
import random

from attenvis import AttentionVis
av = AttentionVis()
def train(model, train_french, train_english, eng_padding_index):
    ####### reshape
    num = len(train_french) // model.batch_size
    # data reshape
    train_french = train_french[0:num*model.batch_size]
    train_english = train_english[0:num*model.batch_size]    
    
    ####### shuffle inputs and labels
    # get index 
    rand_index = tf.random.shuffle(np.array(range(train_english.shape[0])))
    
    # shuffle
    train_F = tf.gather(train_french, rand_index)
    train_E = tf.gather(train_english, rand_index)
    
    ####### Batch
    if train_F.shape[0]%model.batch_size == 0:
        swt = 0
    else:
        swt = 1
            
    # Batch Iterations    
    for i in range((train_F.shape[0]//model.batch_size)+swt):
        ind1 = model.batch_size*i
        ind2 = model.batch_size*i+model.batch_size
        en_X = train_F[ind1:ind2]
        de_X = train_E[ind1:ind2, 0: model.english_window_size - 1]
        de_Y = train_E[ind1:ind2, 1: model.english_window_size]
        mask = np.where(de_Y == eng_padding_index, False, True)
         
        with tf.GradientTape() as tape:
            prob = model.call(en_X, de_X)
            loss = model.loss_function(prob, de_Y, mask)
        grads = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
@av.test_func
def test(model, test_french, test_english, eng_padding_index):
    ####### reshape
    num = len(test_french) // model.batch_size
    # data reshape
    test_french = test_french[0:num*model.batch_size]
    test_english = test_english[0:num*model.batch_size]
    
    ####### Batch
    train_F = test_french
    train_E = test_english
    
    if train_F.shape[0]%model.batch_size == 0:
        swt = 0
    else:
        swt = 1
            
    # Batch Iterations
    perp = 0
    accu = []
    symb = 0
    for i in range((train_F.shape[0]//model.batch_size)+swt):
        ind1 = model.batch_size*i
        ind2 = model.batch_size*i+model.batch_size
        en_X = train_F[ind1:ind2]
        de_X = train_E[ind1:ind2, 0: model.english_window_size - 1]
        de_Y = train_E[ind1:ind2, 1: model.english_window_size]
        mask = np.where(de_Y == eng_padding_index, False, True)
        prob = model.call(en_X, de_X)
        
        symb += np.sum(tf.cast(mask, dtype=tf.float32))
        perp += model.loss_function(prob, de_Y, mask)
        accu.append(model.accuracy_function(prob, de_Y, mask)*np.sum(tf.cast(mask, dtype=tf.float32)))
        
        
        
    perplexity = np.exp(perp / symb)
    accuracy = sum(accu)/symb
    
    return (perplexity, accuracy)

def main():
	
	model_types = {"RNN" : RNN_Seq2Seq, "TRANSFORMER" : Transformer_Seq2Seq}
	if len(sys.argv) != 2 or sys.argv[1] not in model_types.keys():
		print("USAGE: python assignment.py <Model Type>")
		print("<Model Type>: [RNN/TRANSFORMER]")
		exit()

	# Change this to "True" to turn on the attention matrix visualization.
	# You should turn this on once you feel your code is working.
	# Note that it is designed to work with transformers that have single attention heads.
	if sys.argv[1] == "TRANSFORMER":
		av.setup_visualization(enable=False)

	print("Running preprocessing...")
	data_dir   = '../../data'
	file_names = ('fls.txt', 'els.txt', 'flt.txt', 'elt.txt')
	file_paths = [f'{data_dir}/{fname}' for fname in file_names]
	train_eng,test_eng, train_frn,test_frn, vocab_eng,vocab_frn,eng_padding_index = get_data(*file_paths)
	print("Preprocessing complete.")

	model = model_types[sys.argv[1]](FRENCH_WINDOW_SIZE, len(vocab_frn), ENGLISH_WINDOW_SIZE, len(vocab_eng))

	# TODO:
	# Train and Test Model for 1 epoch.
	train(model, train_frn, train_eng, eng_padding_index)
	perplexity, accuracy = test(model, test_frn, test_eng, eng_padding_index)
	print('Perplexity = ' , perplexity)
	print('Accuracy = ', accuracy)

	# Visualize a sample attention matrix from the test set
	# Only takes effect if you enabled visualizations above
	av.show_atten_heatmap()


if __name__ == '__main__':
	main()
