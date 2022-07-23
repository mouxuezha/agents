
from numpy.lib.type_check import real
import tensorflow as tf 
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import math


# LAYER1_SIZE = 500
# LAYER2_SIZE = 500
# LAYER3_SIZE = 500
# LEARNING_RATE = 1e-4 # 1e-3
# TAU = 0.005 # 0.001
# L2 = 0.01

class CriticNetwork:
    """docstring for CriticNetwork"""
    def __init__(self,sess,state_dim,action_dim,**kargs):
        self.time_step = 0
        self.sess = sess
        if 'LAYER_SIZE' in kargs:
            self.LAYER_SIZE = kargs['LAYER_SIZE']
        else:
            self.LAYER_SIZE = [500,500,500] 
        if 'LEARNING_RATE' in kargs:
            self.LEARNING_RATE = kargs['LEARNING_RATE']
        else:
            self.LEARNING_RATE = 1e-4
        if 'TAU' in kargs:
            self.TAU = kargs['TAU']
        else: 
            self.TAU = 0.005
        if 'L2' in kargs:
            self.L2 = kargs['L2']
        else:
            self.L2 = 0.01

        self.config = {'LAYER_SIZE': self.LAYER_SIZE,
         'LEARNING_RATE': self.LEARNING_RATE,
         'TAU' : self.TAU,
         'L2':self.L2}

        # create q network
        self.state_input,\
        self.action_input,\
        self.q_value_output,\
        self.net = self.create_q_network(state_dim,action_dim)

        # create target q network (the same structure with q network)
        self.target_state_input,\
        self.target_action_input,\
        self.target_q_value_output,\
        self.target_update = self.create_target_q_network(state_dim,action_dim,self.net)

        self.create_training_method()

        # initialization 
        self.sess.run(tf.initialize_all_variables())
            
        self.update_target()

    def create_training_method(self):
        # Define training optimizer
        self.y_input = tf.placeholder("float",[None,1])
        weight_decay = tf.add_n([self.L2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output,self.action_input)

    def create_q_network(self,state_dim,action_dim):
        # the layer size could be changed

        state_input = tf.placeholder("float",[None,state_dim])
        action_input = tf.placeholder("float",[None,action_dim])

        N_layers = len(self.LAYER_SIZE)
        W = [] 
        b = []
        real_size = np.append(state_dim,self.LAYER_SIZE)
        real_size = np.append(real_size,np.array([1]))
        for i in range(N_layers):
            # get the omegas 
            if i == 0:
                # before add the action. 
                Wi = self.variable([real_size[i],real_size[i+1]],real_size[i])
            # elif i == N_layers-1:
            else: 
                # which means the lay next to the action. w2 and w3
                Wi = self.variable([real_size[i],real_size[i+1]],real_size[i]+action_dim)
            # get the b
            if i ==1:
                # for b2
                bi = self.variable([real_size[i+1]],real_size[i]+action_dim)
            else:
                bi = self.variable([real_size[i+1]],real_size[i])
            W.append(Wi)
            b.append(bi)
        W_end = tf.Variable(tf.random_uniform([real_size[-2],real_size[-1]],-3e-3,3e-3))
        b_end = tf.Variable(tf.random_uniform([real_size[-1]],-3e-3,3e-3))
        W.append(W_end)
        b.append(b_end)
        Wi_action = self.variable([action_dim,real_size[2]],real_size[1]+action_dim) # w2_action

        # W1 = self.variable([state_dim,layer1_size],state_dim)
        # b1 = self.variable([layer1_size],state_dim)
        # W2 = self.variable([layer1_size,layer2_size],layer1_size+action_dim)
        # W2_action = self.variable([action_dim,layer2_size],layer1_size+action_dim)
        # b2 = self.variable([layer2_size],layer1_size+action_dim)

        # W3 = self.variable([layer2_size,layer3_size],layer2_size+action_dim)
        # b3 = self.variable([layer3_size],layer2_size)
        # # W3_action = self.variable([action_dim,layer3_size],layer2_size+action_dim)
        # # b3 = self.variable([layer3_size],layer2_size+action_dim)
        # ## action has been putted in, so this layer should not put again.

        # W4 = tf.Variable(tf.random_uniform([layer3_size,1],-3e-3,3e-3))
        # b4 = tf.Variable(tf.random_uniform([1],-3e-3,3e-3))

        # automated version
        layer=[0 for i in range(N_layers)] # initiate an list for layers
        for i in range(N_layers):
            if i == 0:
                # the first layer, layer1
                layer[i] = tf.nn.relu(tf.matmul(state_input,W[0]) + b[0])
            elif i == 1:
                # this is to add actions, layer2
                layer[i] = tf.nn.relu(tf.matmul(layer[i-1],W[i]) + tf.matmul(action_input,Wi_action) + b[i])
            else:
                # this is other layers 
                layer[i] = tf.nn.relu(tf.matmul(layer[i-1],W[i])  + b[i])
        q_value_output = tf.identity(tf.matmul(layer[-1],W[-1]) + b[-1])

        # # transition version
        # layer1 = tf.nn.relu(tf.matmul(state_input,W[0]) + b[0])
        # layer2 = tf.nn.relu(tf.matmul(layer1,W[1]) + tf.matmul(action_input,Wi_action) + b[1])
        # # layer3 = tf.nn.relu(tf.matmul(layer2,W3) + tf.matmul(action_input,W3_action) + b3)
        # layer3 = tf.nn.relu(tf.matmul(layer2,W[2])  + b[2])
        # q_value_output = tf.identity(tf.matmul(layer3,W[3]) + b[3])

        # # original version
        # layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
        # layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(action_input,W2_action) + b2)
        # # layer3 = tf.nn.relu(tf.matmul(layer2,W3) + tf.matmul(action_input,W3_action) + b3)
        # layer3 = tf.nn.relu(tf.matmul(layer2,W3)  + b3)
        # q_value_output = tf.identity(tf.matmul(layer3,W4) + b4)

        shuchu_list = [] 
        for i in range(N_layers+1):
            shuchu_list.append(W[i])
            if i ==1:
                # add Wi_action here
                shuchu_list.append(Wi_action)
            shuchu_list.append(b[i])

        return state_input,action_input,q_value_output,shuchu_list

    def create_target_q_network(self,state_dim,action_dim,net):
        state_input = tf.placeholder("float",[None,state_dim])
        action_input = tf.placeholder("float",[None,action_dim])

        ema = tf.train.ExponentialMovingAverage(decay=1-self.TAU) #Exponential Moving Average
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        N_layers = len(self.LAYER_SIZE)
        layers=[0 for i in range(N_layers)] # initiate an list for layers
        for i in range(N_layers):
            if i ==0 :
                # first layer.
                layers[i] = tf.nn.relu(tf.matmul(state_input,target_net[2*i]) + target_net[2*i+1])
            elif i ==1:
                # add action. layer2
                layers[i] = tf.nn.relu(tf.matmul(layers[i-1],target_net[2*i]) + tf.matmul(action_input,target_net[2*i+1]) + target_net[2*i+2])
            else:
                # other layers. layer3 and more.
                layers[i] = tf.nn.relu(tf.matmul(layers[i-1],target_net[2*i+1]) + target_net[2*i+2])

        q_value_output = tf.identity(tf.matmul(layers[-1],target_net[2*N_layers+1]) + target_net[2*N_layers+2])


        # layer1 = tf.nn.relu(tf.matmul(state_input,target_net[0]) + target_net[1])
        # layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + tf.matmul(action_input,target_net[3]) + target_net[4])
        # # layer3 = tf.nn.relu(tf.matmul(layer2,target_net[5]) + tf.matmul(action_input,target_net[6]) + target_net[7])
        # layer3 = tf.nn.relu(tf.matmul(layer2,target_net[5]) + target_net[6])
        # q_value_output = tf.identity(tf.matmul(layer3,target_net[7]) + target_net[8])

        return state_input,action_input,q_value_output,target_update

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,y_batch,state_batch,action_batch):
        self.time_step += 1
        self.sess.run(self.optimizer,feed_dict={
            self.y_input:y_batch,
            self.state_input:state_batch,
            self.action_input:action_batch
            })

    def gradients(self,state_batch,action_batch):
        return self.sess.run(self.action_gradients,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch
            })[0]

    def target_q(self,state_batch,action_batch):
        return self.sess.run(self.target_q_value_output,feed_dict={
            self.target_state_input:state_batch,
            self.target_action_input:action_batch
            })

    def q_value(self,state_batch,action_batch):
        return self.sess.run(self.q_value_output,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch})

    # f fan-in size
    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

    def load_network(self,location):
        location = location + "/saved_critic_networks"
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(location)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path) 
        else:
            print("Could not find old network weights") 

    def save_network(self,location,time_step):
        self.saver=tf.train.Saver(max_to_keep=1)
        print('save critic-network...',time_step) 
        location = location + '/saved_critic_networks/' + 'critic-network'
        self.saver.save(self.sess, location, global_step = time_step)

    def reset_critic_network(self):
        print('MXairfoil: critic network reseted. Say hello to the future' )
        self.sess.run(tf.initialize_all_variables())
        