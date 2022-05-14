import numpy as np
import tensorflow as tf

class RNN_Seq2Seq(tf.keras.Model):
    def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):
        ###### DO NOT CHANGE ##############
        super(RNN_Seq2Seq, self).__init__()
        self.french_vocab_size = french_vocab_size # The size of the French vocab
        self.english_vocab_size = english_vocab_size # The size of the English vocab
        
        self.french_window_size = french_window_size # The French window size
        self.english_window_size = english_window_size # The English window size
        ######^^^ DO NOT CHANGE ^^^##################
        
        # 1) Define any hyperparameters
        self.rnn_size = 35
        self.embedding_size = 50 #32 ~ 256 

        # Define batch size and optimizer/learning rate
        self.batch_size = 100
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.ls = tf.keras.losses.sparse_categorical_crossentropy
        
        # 2) Define embeddings, encoder, decoder, and feed forward layers
        # Embedding:
        self.emb_french = tf.keras.layers.Embedding(self.french_vocab_size, self.embedding_size)
        self.emb_english = tf.keras.layers.Embedding(self.english_vocab_size, self.embedding_size)
        
        # Encoder and Decoder:
        self.encoder = tf.keras.layers.GRU(self.rnn_size, return_sequences=True, return_state=True)
        self.decoder = tf.keras.layers.GRU(self.rnn_size, return_sequences=True, return_state=True)
        
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
        
        # 1) Pass your French sentence embeddings to your encoder
        embed_F = self.emb_french(encoder_input)
        output_1,state_1 = self.encoder(embed_F, initial_state = None)
        
        
        # 2) Pass your English sentence embeddings, and final state of your encoder, to your decoder
        embed_E = self.emb_english(decoder_input)
        output_2,state_2 = self.encoder(embed_E,initial_state = state_1)
        
        # 3) Apply dense layer(s) to the decoder out to generate probabilities
        #dense = self.mlp_relu(output_2)
        prob = self.mlp_softmax(output_2) # if previous is relu change output_2 -> dense

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
        Calculates the total model cross-entropy loss after one forward pass.
        Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.
        
        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        return tf.reduce_sum(tf.boolean_mask(self.ls(labels, prbs), mask))       

