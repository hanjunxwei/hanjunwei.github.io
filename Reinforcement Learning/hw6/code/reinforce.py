import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions
        
        # TODO: Define network parameters and optimizer
        self.state_size = state_size
        
        # Policy Net
        self.reinforcement = tf.keras.Sequential(
            layers=[
                # Input layers
                tf.keras.layers.InputLayer(input_shape=(state_size)),
                # Relu layers
                tf.keras.layers.Dense(190, activation="relu"),
                # Softmax layers
                tf.keras.layers.Dense(num_actions, activation="softmax", name="softmax_layer")
            ], name = "Policy"
        )
        
        # Optimize function
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)

    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        # Go through probability
        prob = self.reinforcement(states)
        # reshape
        prob = tf.reshape(prob, [prob.shape[0],self.num_actions])
        return prob
        

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a Tensorflow scalar
        """
        
        # probability from polic net
        prob = self.call(states)
        
        # indexing
        idx = np.reshape(actions, [len(actions), 1])
        
        # prob of each actions
        p_a = tf.gather_nd(batch_dims=1, params=prob, indices=idx)
        
        # loss value
        loss = - tf.reduce_sum(tf.multiply(tf.math.log(p_a), discounted_rewards))
        return loss

