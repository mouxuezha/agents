# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Flood Sung
# Date: 2016.5.4
# multi-objective change by mouxuezha, 2021.4.16
# try to modify it into 4 dims, mouxuezha,2021.5.27 
# try to make it user-friendly by mouxuezha, 2022.4.8
# -----------------------------------
import gym
import tensorflow as tf
import numpy as np
from ou_noise import OUNoise
# from critic_network import CriticNetwork
# from actor_network_bn import ActorNetwork
from critic_network3 import CriticNetwork
from actor_network_bn3 import ActorNetwork
from replay_buffer import ReplayBuffer

import os
import pickle

class DDPG2:
    """docstring for DDPG"""
    def __init__(self, env,buffer,index,**kargs):
        self.name = 'DDPG_shishi' # name for uploading results
        self.environment = env
        self.real_dim = self.environment.real_dim
        self.performance_dim = self.environment.performance_dim
        # to show which objective is this agent for. 0 and 1 and more.
        self.kind = index 
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        self.sess = tf.InteractiveSession()

        # self.REPLAY_BUFFER_SIZE = 1000000
        # self.REPLAY_START_SIZE = 10000
        self.BATCH_SIZE = 64 
        self.GAMMA = 0.99
        
        # chicun = [400,300]
        if 'REPLAY_START_SIZE' in kargs:
            self.REPLAY_START_SIZE = kargs['REPLAY_START_SIZE']
        else:
            self.REPLAY_START_SIZE = 10000

        if 'chicun' in kargs:
            chicun = kargs['chicun']
        else:
            chicun = [200,300,200]
        if 'LEARNING_RATE_a' in kargs:
            LEARNING_RATE_a = kargs['LEARNING_RATE_a']
        else:
            LEARNING_RATE_a = 0.25e-4 
        if 'LEARNING_RATE_c' in kargs:
            LEARNING_RATE_c = kargs['LEARNING_RATE_c']
        else:
            LEARNING_RATE_c = 0.25e-3
        if 'TAU' in kargs:
            TAU = kargs['TAU']
        else:
            TAU = 0.0005
        if 'L2' in kargs:
            L2 = kargs['L2']
        else:
            L2 = 0.01 

        self.actor_network = ActorNetwork(self.sess,self.state_dim,self.action_dim,self.kind,LAYER_SIZE=chicun,LEARNING_RATE=LEARNING_RATE_a,TAU = TAU)
        self.critic_network = CriticNetwork(self.sess,self.state_dim,self.action_dim,LAYER_SIZE = chicun ,LEARNING_RATE = LEARNING_RATE_c,TAU = TAU,L2 = L2)

        # initialize replay buffer
        # self.replay_buffer = ReplayBuffer(self.REPLAY_BUFFER_SIZE)
        self.replay_buffer = buffer 

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

        # self.exploration_noise = simpleNoise(self.action_dim,0.6,-1,1)
        self.ave_reward_save0 = np.array([0.0,0.0]).reshape(1,2)
        # self.raw_state_save = np.zeros(self.real_dim,dtype=float).reshape(1,self.real_dim) # this is what it should be
        self.raw_state_save = np.ones(self.real_dim,dtype=float).reshape(1,self.real_dim)*0.5 # this is toulan for Demo180
        self.raw_state_episode = np.array([0]).reshape(1,1)
        # self.raw_state_performance = np.array([0.6,1.05,30.0,0.5]).reshape(1,4)
        self.raw_state_performance = np.ones(self.performance_dim)
        self.raw_state_performance = self.raw_state_performance * 0.2 
        self.raw_state_performance = self.raw_state_performance.reshape(1,self.performance_dim)
        # these for stop and continue.
        
        # this is to synchronize the raw_state, performance and so on in agent and in environment.
        # self.environment.extra_set(raw_state_performance = self.raw_state_performance,raw_state_save=self.raw_state_save)

        self.episode_last_ave = np.array([0]).reshape(1,1)
        self.episode_last_gol = np.array([0]).reshape(1,1)

    def train(self):
        #print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(self.BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        # shishi3 = np.asarray([data[2] for data in minibatch])
        # shishi3 =  np.asarray([data[3][self.action_dim+self.kind] for data in minibatch])
        # shishi3 =  np.asarray([data[3][self.state_dim-1+self.kind] for data in minibatch])
        # reward_batch = shishi3
        reward_batch = np.asarray([data[2] for data in minibatch])


        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch,[self.BATCH_SIZE,self.action_dim])

        # Calculate y_batch

        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch,next_action_batch)
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + self.GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch,[self.BATCH_SIZE,1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch,state_batch,action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch,action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch,state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def noise_action(self,state):
        # Select action a_t according to the current policy and exploration noise
        # print('state:',state)
        # print('state_dim:',self.state_dim)
        action = self.actor_network.action(state)
        # print('action:',action)

        
        return action+self.exploration_noise.noise()

    def action(self,state):
        action = self.actor_network.action(state)
        return action

    def perceive(self,state,action,reward,next_state,done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state,action,reward,next_state,done)
        # if self.kind == 1 :
        #     self.replay_buffer.add(state,action,reward,next_state,done)
            # if there are more than one agent, only add one time.

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() >  self.REPLAY_START_SIZE:
            self.train()

        #if self.time_step % 10000 == 0:
            #self.actor_network.save_network(self.time_step)
            #self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()

    def save_agent(self,location,N):
        
        if not(os.path.exists(location)):
            #which means there are no such folder, then mkdir.
            try:
                os.mkdir(location)
            except:
                print('MXairfoil: can not make dir for saveing agent. ',location)
                location = ''
            
        self.actor_network.save_network(location,N)
        self.critic_network.save_network(location,N)

        wenjianming = location + '/ave_reward_save'+str(self.kind)+'.pkl'
        pickle.dump(self.ave_reward_save0,open(wenjianming,'wb'))

        wenjianming = location + '/raw_state_save'+str(self.kind)+'.pkl'
        pickle.dump(self.raw_state_save,open(wenjianming,'wb'))

        wenjianming = location + '/raw_state_episode'+str(self.kind)+'.pkl'
        pickle.dump(self.raw_state_episode,open(wenjianming,'wb'))

        wenjianming = location + '/raw_state_performance'+str(self.kind)+'.pkl'
        pickle.dump(self.raw_state_performance,open(wenjianming,'wb'))

        print('MXairfoil: tried to save the agent in ',location)
        print('En Taro XXH!')

    def load_agent(self,location):
        # self.actor_network.load_network(location)
        # self.critic_network.load_network(location)
        try:
            self.actor_network.load_network(location)
            self.critic_network.load_network(location)
        except Exception as e :
            print('MXairfoil: it seems no prepared networks to load.')

        wenjianming = location + '/ave_reward_save0.pkl'
        try:
            self.ave_reward_save0 = pickle.load(open(wenjianming,'rb'))
        except:
            print('MXairfoil: no prepared ave_reward_save0 there')
        chicun = self.ave_reward_save0.shape
        self.episode_last_ave = self.ave_reward_save0[chicun[0]-1][0]
        
        wenjianming = location + '/raw_state_save'+str(self.kind)+'.pkl'
        try:
            self.raw_state_save = pickle.load(open(wenjianming,'rb'))
        except:
            print('MXairfoil: no prepared raw_state_save there')
        chicun = self.raw_state_save.shape
        if chicun[0] > 1:
            # which means there are real state recorded
            for i in range(chicun[0]):
               self.environment.set_raw_state(self.raw_state_save[i].reshape(chicun[1],))
        else:
            # which means no real state recorded, only a initail one[0,0,0...] 
            print('MXairfoil: no raw_state recorded in fact.')

        wenjianming = location + '/raw_state_episode'+str(self.kind)+'.pkl'
        try:
            self.raw_state_episode = pickle.load(open(wenjianming,'rb'))
        except:
            print('MXairfoil: no prepared raw_state_episode there')
        chicun = self.raw_state_episode.shape
        self.episode_last_gol = self.raw_state_episode[chicun[0]-1].reshape(1,1) 

        wenjianming = location + '/raw_state_performance'+str(self.kind)+'.pkl'
        try:
            self.raw_state_performance = pickle.load(open(wenjianming,'rb'))
        except:
            print('MXairfoil: no prepared raw_state_performance there')

    def feed_buffer(self,buffer):
        #feed buffer into agent
        size1 = self.replay_buffer.size()
        try:
            size2 = buffer.size()
        except:
            size2 = 0 
            print('MXairfoil: something wrong when feed_buffer in agent. maybe it is not replay buffer.')
        if size1==size2:
            self.replay_buffer = buffer
            print('MXairfoil: successfully load the buffer into agent')
        
    def save_buffer(self,wenjianjia):
        #buffer.save_buffer('C:/Users/y/Desktop/DDPGshishi/agent_sin2_buffer.pkl')
        # wenjianming = wenjianjia + '/agent' + str(self.kind) + '_shishi2_buffer.pkl'
        wenjianming = wenjianjia 
        self.replay_buffer.save_buffer(wenjianming)

    def record_ave_reward(self,episode,ave_reward0):
        # chicun = self.ave_reward_save0.shape
        self.ave_reward_save0 = np.append(self.ave_reward_save0,np.array([episode+self.episode_last_ave,ave_reward0]).reshape(1,2),axis=0)

    def record_raw_state(self,state,**kargs):
        self.raw_state_save = np.append(self.raw_state_save,state.reshape(1,len(state)),axis=0)
        if 'episode' in kargs:
            self.raw_state_episode = np.append(self.raw_state_episode,np.array(kargs['episode']).reshape(1,1)+self.episode_last_gol,axis=0)
        if 'performance' in kargs:
            # self.raw_state_performance = np.append(self.raw_state_performance,np.array(kargs['performance']).reshape(1,4),axis=0)
            self.raw_state_performance = np.append(self.raw_state_performance,np.array(kargs['performance']).reshape(1,self.performance_dim),axis=0)
            

    def reset_network(self):
        print('MXairfoil: reset networks in agent ' + str(self.kind))
        self.actor_network.reset_actor_network()
        self.critic_network.reset_critic_network()
    
    def reset_buffer(self):
        print('MXairfoil: reset buffer in agent ' + str(self.kind))
        self.replay_buffer.erase()

    def reset_agent(self,**kargs):
        print('MXairfoil: reset networks in agent ' + str(self.kind))
        self.actor_network.reset_actor_network()
        self.critic_network.reset_critic_network()
        self.ave_reward_save0 = np.array([0,0]).reshape(1,2)
        self.reset_buffer()
        self.environment.N_artificial_tip = 0 
        self.environment.N_step=0
        self.random_r = 0.01
        if 'model' in kargs:
            if kargs['model'] == 'all':
                pass


        # these things about high-order should be reseted.
        state_0 = np.zeros(self.real_dim,dtype=float)
        # self.raw_state_save = np.array([0.0,0.0]).reshape(1,self.real_dim)
        self.raw_state_save = state_0.reshape(1,self.real_dim)
        self.raw_state_episode = np.array([0]).reshape(1,1)

        # self.raw_state_performance = np.array([0.6,1.05,30.0,0.5]).reshape(1,4)
        self.raw_state_performance = np.ones(self.performance_dim)
        self.raw_state_performance = self.raw_state_performance * 0.5 
        self.raw_state_performance = self.raw_state_performance.reshape(1,self.performance_dim)

        if 'location' in kargs:
            import shutil
            location = kargs['location']
            try:
                print('MXairfoil: delet legacy files')
                shutil.rmtree(location+'/saved_critic_networks')
                shutil.rmtree(location+'/saved_actor_networks')
                os.remove(location + '/ave_reward_save'+str(self.kind)+'.pkl')
                os.remove(location + '/raw_state_episode'+str(self.kind)+'.pkl')
                os.remove(location + '/raw_state_performance'+str(self.kind)+'.pkl')
                os.remove(location + '/raw_state_save'+str(self.kind)+'.pkl')
                os.remove(location + '/buffer_'+str(self.real_dim)+'dim'+'.pkl')
            except:
                print('MXairfoil: fail to delet legacy files')

        # self.save_buffer()

    def get_config(self):
        actor_config = self.actor_network.config
        critic_config = self.critic_network.config
        return [actor_config,critic_config]
    