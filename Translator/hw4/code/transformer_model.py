import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

from attenvis import AttentionVis

av = AttentionVis()

class Transformer_Seq2Seq(tf.keras.Model):
    def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

        ######vvv DO NOT CHANGE vvv##################
        super(Transformer_Seq2Seq, self).__init__()

        self.french_vocab_size = french_vocab_size # The size of the French vocab
        self.english_vocab_size = english_vocab_size # The size of the English vocab

        self.french_window_size = french_window_size # The French window size
        self.english_window_size = english_window_size # The English window size
        ######^^^ DO NOT CHANGE ^^^##################



        
        # 1) Define any hyperparameters
        self.embedding_size = 50 #32 ~ 256 

        # Define batch size and optimizer/learning rate
        self.batch_size = 100
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.ls = tf.keras.losses.sparse_categorical_crossentropy
        
        # 2) Define embeddings, encoder, decoder, and feed forward layers
        # Define English and French embedding layers:
        #self.emb_french = tf.keras.layers.Embedding(self.french_vocab_size, self.embedding_size)
        #self.emb_english = tf.keras.layers.Embedding(self.english_vocab_size, self.embedding_size)
        self.emb_french = tf.Variable(tf.random.truncated_normal(shape=[self.french_vocab_size, self.embedding_size], mean=0, stddev=0.01))
        self.emb_english = tf.Variable(tf.random.truncated_normal(shape=[self.english_vocab_size, self.embedding_size], mean=0, stddev=0.01))
        
        # Create positional encoder layers
        self.position_F = transformer.Position_Encoding_Layer(self.french_window_size, self.embedding_size)
        self.position_E = transformer.Position_Encoding_Layer(self.english_window_size, self.embedding_size)
        
        # Encoder and Decoder:
        self.encoder = transformer.Transformer_Block(self.embedding_size, is_decoder=False, multi_headed = False)
        self.decoder = transformer.Transformer_Block(self.embedding_size, is_decoder=True, multi_headed = False)
        
        # Dense Layer:
        #self.mlp_relu = tf.keras.layers.Dense(units = 80, activation='relu')
        self.mlp_softmax = tf.keras.layers.Dense(self.english_vocab_size, activation='softmax')        
    @tf.function
    def call(self, encoder_input, decoder_input):
        """
        :param encoder_input: batched ids corresponding to French sentences
        :param decoder_input: batched ids corresponding to English sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """

        # TODO:
        #1) Add the positional embeddings to French sentence embeddings
        #embed_F = self.position_F(self.emb_french(encoder_input))
        embed_F = tf.nn.embedding_lookup(self.emb_french,encoder_input)
        
        #2) Pass the French sentence embeddings to the encoder
        output_1 = self.encoder(embed_F)
        
        #3) Add positional embeddings to the English sentence embeddings
        #embed_E = self.position_E(self.emb_english(decoder_input))
        embed_E = tf.nn.embedding_lookup(self.emb_english,decoder_input)
        
        #4) Pass the English embeddings and output of your encoder, to the decoder
        output_2 = self.decoder(embed_E, context = output_1)
        
        #5) Apply dense layer(s) to the decoder out to generate probabilities
        prob = self.mlp_softmax(output_2)
        return prob

    def accuracy_function(self, prbs, labels, mask):
        """
        DO NOT CHANGE
        Computes the batch accuracy

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """

        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
        return accuracy


    def loss_function(self, prbs, labels, mask):
        """
        Calculates the model cross-entropy loss after one forward pass
        Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.
        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        # Note: you can reuse this from rnn_model.

        return tf.reduce_sum(tf.boolean_mask(self.ls(labels, prbs), mask))       
    
    @av.call_func
    def __call__(self, *args, **kwargs):
        return super(Transformer_Seq2Seq, self).__call__(*args, **kwargs)
