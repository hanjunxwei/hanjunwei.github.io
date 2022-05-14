import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.
        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # hyperparameters

        self.vocab_size = vocab_size
        self.batch_size = 10
        self.window_size = 20
        self.embedding_size = 30
        self.rnn_size = 35     ## hidden state size
        
        # Parameter
        ## dense Layer
        self.mlp_relu = tf.keras.layers.Dense(units = 80, activation='relu')
        self.mlp_softmax = tf.keras.layers.Dense(units = vocab_size, activation='softmax')
        ## RNN
        self.LSTM = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        self.GRU = tf.keras.layers.GRU(self.rnn_size, return_sequences=True, return_state=True)
        
        # Function
        self.emb = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.ls = tf.keras.losses.sparse_categorical_crossentropy
 

    def call(self, inputs, initial_state = None):
        """
        - You must use an embedding layer as the first layer of your network
        (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state
        (NOTE 1: If you use an LSTM, the final_state will be the last two RNN outputs,
        NOTE 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU
        """
        
        embedded = self.emb(inputs)
        #output, state1, state2 = self.LSTM(embedded, initial_state=initial_state)
        output, state = self.GRU(embedded, initial_state = initial_state)
        
        dense = self.mlp_relu(output)
        prob = self.mlp_softmax(dense)

        #return prob,(state1,state2)
        return prob, state


    def loss(self, probabilities, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probabilities: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the average loss of the model as a tensor of size 1
        """

        #TODO: Fill in
        # We recommend using tf.keras.losses.sparse_categorical_crossentropy

        return tf.reduce_mean(self.ls(labels, probabilities))


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples (remember to batch!)
    Here you will also want to reshape your inputs and labels so that they match
    the inputs and labels shapes passed in the call and loss functions respectively.
    
    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    ## reshape
    num = len(train_inputs) // model.window_size
    
    ## data reshape
    train_inputs = train_inputs[0:num*model.window_size].reshape(-1,model.window_size)
    train_labels = train_labels[0:num*model.window_size].reshape(-1,model.window_size)
    
    

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
            prob = model.call(X)[0]
            loss = model.loss(prob,Y)
        grads = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples (remember to batch!)
    Here you will also want to reshape your inputs and labels so that they match
    the inputs and labels shapes passed in the call and loss functions respectively.

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """

    ## Reshape
    initial_state = None
    num = len(test_inputs) // model.window_size
    ## data reshape
    test_inputs = test_inputs[0:num*model.window_size].reshape(-1,model.window_size)
    test_labels = test_labels[0:num*model.window_size].reshape(-1,model.window_size)
    

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
        prob = model.call(X, initial_state)[0]
        temp.append(tf.reduce_mean(model.loss(prob, Y)))#state
    
    return np.exp(np.mean(temp))

def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    # NOTE: Feel free to play around with different sample_n values

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0,0,:])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n])/np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n,p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    ## read data
    test_file = "../../data/test.txt"
    train_file = "../../data/train.txt"
    
    train_data, test_data, dictionary = get_data(train_file,test_file)
    
    ## drop front and end
    inputs_tr = train_data[0:len(train_data)-1]
    labels_tr = train_data[1:len(train_data)]
    inputs_te = test_data[0:len(test_data)-1]
    labels_te = test_data[1:len(test_data)]
    vocab_size = len(dictionary)
    RNN = Model(vocab_size)
    
    # Set-up the training step
    train(RNN,inputs_tr,labels_tr)
    
    # Set up the testing steps
    result = test(RNN, inputs_te, labels_te)
    
    # Print out perplexity
    print(result)
    
    # Try printing out various sentences with different start words and sample_n parameters
    
if __name__ == '__main__':
    main()
