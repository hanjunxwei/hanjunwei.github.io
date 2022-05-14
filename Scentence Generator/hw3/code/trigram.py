import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from functools import reduce
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # initialize emnbedding_size, batch_size, and any other hyperparameters

        self.vocab_size = vocab_size
        self.embedding_size = 30
        self.batch_size = 200

        # initialize embeddings and forward pass weights (weights, biases)
        
        ## Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
        
        ## random initilizer
        def create_variable(dims):
            return tf.Variable(tf.random.truncated_normal(dims, stddev=.1, dtype=tf.float32))  ## tf.float32
        
        ## Learnable parameter
        
        self.look_up = tf.nn.embedding_lookup
        self.loss = tf.keras.losses.sparse_categorical_crossentropy
        self.relu = tf.nn.relu
        
        self.E = create_variable([self.vocab_size, self.embedding_size])
        self.w1 = create_variable([2*self.embedding_size, self.vocab_size])
        self.b1 = create_variable([self.vocab_size])
        self.w2 = create_variable([self.vocab_size, self.vocab_size])
        self.b2 = create_variable([self.vocab_size])   
        

    def call(self, inputs):
        """
        You must use an embedding layer as the first layer of your network
        (i.e. tf.nn.embedding_lookup)

        :param inputs: word ids of shape (batch_size, 2)
        :return: probabilities: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """
        embedded = tf.concat([self.look_up(self.E,inputs[:,0]),self.look_up(self.E,inputs[:,1])],1)
        dense1 = self.relu(embedded @ self.w1 + self.b1)
        dense2 = dense1 @ self.w2 + self.b2
        prob = tf.nn.softmax(dense2)
        return prob
        


    def loss_function(self, probabilities, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probabilities: a matrix of shape (batch_size, vocab_size)
        :return: the average loss of the model as a tensor of size 1
        """
        return tf.reduce_mean(self.loss(labels, probabilities))

    
def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.
    Remember to shuffle your inputs and labels - ensure that they are shuffled
    in the same order. Also you should batch your input and labels here.

    :param model: the initilized model to use for forward and backward pass
    :param train_input: train inputs (all inputs for training) of shape (num_inputs,2)
    :param train_labels: train labels (all labels for training) of shape (num_inputs,)
    :return: None
    """
    # shuffle inputs and labels
    ## get index 
    rand_index = tf.random.shuffle(np.array(range(train_labels.shape[0])))
    
    ## shuffle
    train_inputs = tf.gather(train_inputs, rand_index)
    train_labels = tf.gather(train_labels, rand_index)
    
    # Batch
    if train_inputs.shape[0]%model.batch_size == 0:
        swt = 0
    else:
        swt = 1
            
    # Batch Iterations    
    for i in range((train_inputs.shape[0]//model.batch_size)+swt):
        ind1 = model.batch_size*i
        ind2 = model.batch_size*i+model.batch_size
        X = train_inputs[ind1:ind2]
        Y = train_labels[ind1:ind2]
         
        with tf.GradientTape() as tape:
            prob = model.call(X)
            loss = model.loss_function(prob,Y)
        grads = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    
def test(model, test_inputs, test_labels):
    """
    Runs through all test examples. Test input should be batched here.

    :param model: the trained model to use for prediction
    :param test_input: train inputs (all inputs for testing) of shape (num_inputs,2)
    :param test_input: train labels (all labels for testing) of shape (num_inputs,)
    :returns: perplexity of the test set
    """

    # TODO: Fill in
    # NOTE: Ensure a correct perplexity formula (different from raw loss)
    # Batch
    if test_inputs.shape[0]%model.batch_size == 0:
        swt = 0
    else:
        swt = 1
    temp = []
    # Batch Iterations    
    for i in range((test_inputs.shape[0]//model.batch_size)+swt):
        ind1 = model.batch_size*i
        ind2 = model.batch_size*i+model.batch_size
        X = test_inputs[ind1:ind2]
        Y = test_labels[ind1:ind2]
        temp.append(tf.reduce_mean(model.loss_function(model.call(X), Y)))
    
    return np.exp(np.mean(temp))


def generate_sentence(word1, word2, length, vocab, model):
    """
    Given initial 2 words, print out predicted sentence of targeted length.

    :param word1: string, first word
    :param word2: string, second word
    :param length: int, desired sentence length
    :param vocab: dictionary, word to id mapping
    :param model: trained trigram model

    """

    # NOTE: This is a deterministic, argmax sentence generation

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    output_string = np.zeros((1, length), dtype=np.int)
    output_string[:, :2] = vocab[word1], vocab[word2]

    for end in range(2, length):
        start = end - 2
        output_string[:, end] = np.argmax(
            model(output_string[:, start:end]), axis=1)
    text = [reverse_vocab[i] for i in list(output_string[0])]

    print(" ".join(text))


def main():
    # TODO: Pre-process and vectorize the data using get_data from preprocess
    test_file = "../../data/test.txt"
    train_file = "../../data/train.txt"
    
    train_data, test_data, dictionary = get_data(train_file,test_file)

    # TO-DO:  Separate your train and test data into inputs and labels
    inputs_tr = np.concatenate((train_data[0:len(train_data)-2].reshape(-1,1),train_data[1:len(train_data)-1].reshape(-1,1)),1)
    labels_tr = train_data[2:len(train_data)]

    inputs_te = np.concatenate((test_data[0:len(test_data)-2].reshape(-1,1),test_data[1:len(test_data)-1].reshape(-1,1)),1)
    labels_te = test_data[2:len(test_data)]
    
    # TODO: initialize model
    trigram = Model(len(dictionary))

    # TODO: Set-up the training step
    train(trigram,inputs_tr,labels_tr)

    # TODO: Set up the testing steps
    result = test(trigram, inputs_te, labels_te)
    # Print out perplexity
    print(result)

    # BONUS: Try printing out sentences with different starting words
    generate_sentence("a", "person", 20, dictionary, trigram)
    generate_sentence("may", "be", 20, dictionary, trigram)
    

if __name__ == '__main__':
    main()
