import tensorflow.compat.v1 as tf
import numpy as np
import gymnasium as gym  #requires OpenAI gym installed
env = gym.envs.make("MountainCarContinuous-v0") 

tf.disable_v2_behavior()
tf.reset_default_graph()

input_dims = 2
state_placeholder = tf.placeholder(tf.float32, [None, input_dims]) 

def value_function(state):
    n_hidden1 = 400  
    n_hidden2 = 400
    n_outputs = 1
    
    with tf.variable_scope("value_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()
        
        hidden1 = tf.layers.dense(state, n_hidden1, tf.nn.elu, init_xavier)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu, init_xavier) 
        V = tf.layers.dense(hidden2, n_outputs, None, init_xavier)
    return V


def policy_network(state):
    n_hidden1 = 40
    n_hidden2 = 40
    n_outputs = 1
    
    with tf.variable_scope("policy_network"):
        init_xavier = tf.contrib.layers.xavier_initializer()
        
        hidden1 = tf.layers.dense(state, n_hidden1, tf.nn.elu, init_xavier)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, tf.nn.elu, init_xavier)
        mu = tf.layers.dense(hidden2, n_outputs, None, init_xavier)
        sigma = tf.layers.dense(hidden2, n_outputs, None, init_xavier)
        sigma = tf.nn.softplus(sigma) + 1e-5
        norm_dist = tf.contrib.distributions.Normal(mu, sigma)
        action_tf_var = tf.squeeze(norm_dist.sample(1), axis=0)
        action_tf_var = tf.clip_by_value(
            action_tf_var, env.action_space.low[0], 
            env.action_space.high[0])
    return action_tf_var, norm_dist

################################################################
#sample from state space for state normalization
import sklearn
import sklearn.preprocessing
                                    
state_space_samples = np.array(
    [env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(state_space_samples)

#function to normalize states
def scale_state(state):                 #requires input shape=(2,)
    scaled = scaler.transform([state])
    return scaled                       #returns shape =(1,2)   
###################################################################

lr_actor = 0.00002  #set learning rates
lr_critic = 0.001

# define required placeholders
action_placeholder = tf.placeholder(tf.float32)
delta_placeholder = tf.placeholder(tf.float32)
target_placeholder = tf.placeholder(tf.float32)

action_tf_var, norm_dist = policy_network(state_placeholder)
V = value_function(state_placeholder)

# define actor (policy) loss function
loss_actor = -tf.log(norm_dist.prob(action_placeholder) + 1e-5) * delta_placeholder
training_op_actor = tf.train.AdamOptimizer(
    lr_actor, name='actor_optimizer').minimize(loss_actor)

# define critic (state-value) loss function
loss_critic = tf.reduce_mean(tf.squared_difference(
                             tf.squeeze(V), target_placeholder))
training_op_critic = tf.train.AdamOptimizer(
        lr_critic, name='critic_optimizer').minimize(loss_critic)
################################################################
#Training loop
gamma = 0.99        #discount factor
num_episodes = 300

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    episode_history = []
    for episode in range(num_episodes):
        #receive initial state from E
        state = env.reset()   # state.shape -> (2,)
        reward_total = 0 
        steps = 0
        done = False
        while (not done):
                
            #Sample action according to current policy
            #action.shape = (1,1)
            action  = sess.run(action_tf_var, feed_dict={
                          state_placeholder: scale_state(state)})
            #Execute action and observe reward & next state from E
            # next_state shape=(2,)    
            #env.step() requires input shape = (1,)
            next_state, reward, done, _ = env.step(
                                    np.squeeze(action, axis=0)) 
            steps +=1
            reward_total += reward
            #V_of_next_state.shape=(1,1)
            V_of_next_state = sess.run(V, feed_dict = 
                    {state_placeholder: scale_state(next_state)})  
            #Set TD Target
            #target = r + gamma * V(next_state)     
            target = reward + gamma * np.squeeze(V_of_next_state) 
            
            # td_error = target - V(s)
            #needed to feed delta_placeholder in actor training
            td_error = target - np.squeeze(sess.run(V, feed_dict = 
                        {state_placeholder: scale_state(state)})) 
            
            #Update actor by minimizing loss (Actor training)
            _, loss_actor_val  = sess.run(
                [training_op_actor, loss_actor], 
                feed_dict={action_placeholder: np.squeeze(action), 
                state_placeholder: scale_state(state), 
                delta_placeholder: td_error})
            #Update critic by minimizinf loss  (Critic training)
            _, loss_critic_val  = sess.run(
                [training_op_critic, loss_critic], 
                feed_dict={state_placeholder: scale_state(state), 
                target_placeholder: target})
            
            state = next_state
            #end while
        episode_history.append(reward_total)
        print("Episode: {}, Number of Steps : {}, Cumulative reward: {:0.2f}".format(
            episode, steps, reward_total))
        
        if np.mean(episode_history[-100:]) > 90 and len(episode_history) >= 101:
            print("****************Solved***************")
            print("Mean cumulative reward over 100 episodes:{:0.2f}" .format(
                np.mean(episode_history[-100:])))