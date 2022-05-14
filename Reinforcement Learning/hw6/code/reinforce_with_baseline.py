import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions
        self.state_size    = state_size
        
        # actor
        self.reinforcement = tf.keras.Sequential(
            layers=[
                tf.keras.layers.InputLayer(input_shape=(state_size)),
                tf.keras.layers.Dense(120, activation="relu"),
                tf.keras.layers.Dense(num_actions, activation="softmax", name="softmax_layer")
            ], name="actor")
        
        
        # critic
        self.critic = tf.keras.Sequential(
            layers=[
                tf.keras.layers.InputLayer(input_shape=(state_size)),
                tf.keras.layers.Dense(120, activation="relu"),
                tf.keras.layers.Dense(1),
            ], name="critic")
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.001)

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
        
        prob = self.reinforcement(states)
        p_shape = [prob.shape[0], self.num_actions]
        prob = tf.reshape(prob, p_shape)
        return prob
        


    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode.
        :return: A [episode_length] matrix representing the value of each state.
        """

        value = self.critic(states)
        return tf.squeeze(value)

    
    
    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Refer to the lecture slides referenced in the handout to see how this is done.

        Remember that the loss is similar to the loss as in part 1, with a few specific changes.

        1) In your actor loss, instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply with your advantage. 
        See handout/slides for definition of advantage.
        
        2) In your actor loss, you must use tf.stop_gradient on the advantage to stop the loss calculated on the actor network 
        from propagating back to the critic network.
        
        3) See handout/slides for how to calculate the loss for your critic network.

        :param states: A batch of states of shape (episode_length, state_size)
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        idx_shape = [len(actions), 1]
        
        prob      = self.call(states)
        idx       = np.reshape(actions, idx_shape)
        p_a       = tf.gather_nd(batch_dims=1, params=prob, indices=idx)
        value     = self.value_function(states)
        advantage = tf.stop_gradient(tf.math.subtract(discounted_rewards, value))
        loss_act  = - tf.reduce_sum(tf.multiply(tf.math.log(p_a),advantage))
        loss_crt  = tf.reduce_sum(tf.math.square(tf.math.subtract(discounted_rewards, value)))
        loss      = loss_crt + loss_act
        return loss
