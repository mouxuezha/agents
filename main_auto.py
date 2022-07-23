# from unittest import result
import filter_env
# from ddpg2 import *
from ddpg_shishi import *
import gc
gc.enable()
from gym import wrappers
import os
import shutil
from huatu import huatu
from huatu import Pjudge
from time_ratio import time_ratio
import time
from multiprocessing import Process
from transfer import transfer
import random
import sys 

class auto_jisuan():
    def __init__(self, **kargs):
        print('MXairfoil:shoubuliao le cao')
        
        self.N_threads = 2 
        self.EPISODES = 1000 #10000
        self.REPLAY_BUFFER_SIZE = 1000000 
        self.REPLAY_START_SIZE = 10000
        self.jishiqi = time_ratio(zhonglei_list = ['train_ANN','surrogate'])
        

        # some default value to avoid GG. They are not nessarry in fact.
        self.ENV_NAME = 'NACA652_env-v0'
        self.TEST = 10
        self.dx = 0.1
        # dx = 0 
        self.real_dim =2 
        self.ave_reward_threshold = 85 
        self.feed_back_enable = 0 
        self.N_steps = 100 
        
        self.chicun = [200,300,200]
        self.LEARNING_RATE_a = 0.25e-4 
        self.LEARNING_RATE_c = 0.25e-3
        self.TAU = 0.0005
        self.L2 = 0.01 

        if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
            # which means in my diannao.
            self.work_location = 'C:/Users/y/Desktop/DDPGshishi/agents'
            self.storage_location = 'E:/EnglishMulu/agents'
        elif os.environ['COMPUTERNAME'] == 'DESKTOP-132CR84' :
            # which means in new working zhan.
            # D:\XXHcode\DDPGshishi
            self.work_location = 'D:/XXHcode/DDPGshishi/agents'
            self.storage_location = 'E:/XXHdatas/EnglishMulu/agents'
        else:
            #which means in 106 server
            self.work_location = 'C:/Users/106/Desktop/DDPGshishi/agents'
            self.storage_location = 'E:/XXHdatas/agents'
        self.kaiguan_location = self.work_location+ '/新建文本文档.txt'
        self.shijian = time.strftime("%Y-%m-%d", time.localtime())

        if 'ENV_NAME' in kargs:
            self.ENV_NAME = kargs['ENV_NAME' ]
        if 'dx' in kargs:
            self.dx = kargs['dx' ]   
        if 'constrain_flag' in kargs:
            self.constrain_flag = kargs['constrain_flag']
        else:
            self.constrain_flag = False
        # then set the cases
        if self.ENV_NAME == 'NACA652_env-v0':
            self.set_case_NACA65() 
        elif self.ENV_NAME == 'Demo180_env-v0':
            self.set_case_Demo180() 
        elif self.ENV_NAME == 'Rotor67_env-v0':
            self.set_case_Rotor67()
        elif self.ENV_NAME == 'CDA42_env-v0':
            self.set_case_CDA2d_unconstrained()
        elif self.ENV_NAME == 'CDA43_env-v0':
            self.set_case_CDA2d_constrained()
        elif self.ENV_NAME == 'CDA44_env-v0':
            self.set_case_CDA4d()
        else:
            raise Exception('MXairfoil: G! Invalid case for auto_jisuan')

        if 'ave_reward_threshold' in kargs:
            self.ave_reward_threshold = kargs['ave_reward_threshold' ]
        if 'feed_back_enable' in kargs:
            self.feed_back_enable = kargs['feed_back_enable' ]
        # if 'dx' in kargs: # this should be set before setting case.
        #     self.dx = kargs['dx' ]
        # if 'dim' in kargs: # this should be determined by 
        #     self.real_dim = kargs['dim' ]
        if 'buffer_reuse' in kargs:
            self.buffer_reuse =kargs['buffer_reuse']
        else:
            self.buffer_reuse = False
        if 'EPISODES' in kargs :
            self.EPISODES = kargs['EPISODES']
        if 'agent_reuse' in kargs:
            self.agent_reuse = kargs['agent_reuse']
        else:
            self.agent_reuse = False
        if 'zuobi' in kargs:
            self.zuobi = kargs['zuobi']
        else:
            self.zuobi = False

        if 'episode_min' in kargs:
            self.episode_min = kargs['episode_min']
        else:
            self.episode_min = 0 

        self.tiaochu = 0

        self.init_agents_flag= False
        self.converge_history = [] # this is to record converge_history 
        self.converge_history_opt = [] # this is to record detected optimization.  

        if self.jishiqi:
            self.env.step = self.jishiqi.jishi_type_decorate('surrogate',self.env.step)


    def init_agents(self):
        # separate from self.__init__

        self.buffer = ReplayBuffer(self.REPLAY_BUFFER_SIZE,state_dim=self.state_dim,action_dim = self.real_dim)
        self.agent1 = DDPG2(self.env,self.buffer,1,chicun = self.chicun)
        # # this is sacrifice for saveing agent0 successfully, which is xuanxue and I don't konw why.
        self.agent0 = DDPG2(self.env,self.buffer,0,chicun = self.chicun)
        # agent1 = DDPG2(env,buffer,1)
         
        if self.buffer_reuse:
            self.buffer.load_buffer(self.buffer_location)
        if self.agent_reuse:
            try:
                self.agent0.load_agent(self.agent0_location)
            except Exception as e:
                print('MXairfoil: no prepared agent0 there ')
                print(e)
                raw_state_new = self.env.reset_random()*1
                self.agent0.record_raw_state(raw_state_new[0:self.real_dim])
        
        if self.jishiqi:
            print('MXairfoil: jishiqi established for agent')
            self.agent0.perceive = self.jishiqi.jishi_type_decorate('train_ANN',self.agent0.perceive)

        self.init_agents_flag = True 

    def set_case_NACA65(self):
        if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
            # which means in my diannao.
            sys.path.append(r'C:/Users/y/Desktop/DDPGshishi/DDPG-master/NACA652-gym/KrigingPython')
        else:
            sys.path.append(r'C:/Users/106/Desktop/DDPGshishi/DDPG-master/NACA652-gym/KrigingPython')
        from Surrogate_01de import Surrugate
        from Surrogate_01de import record_progress # it must be here, or class record_progress can not be found.
        import NACA652_gym
        self.real_obs_space_l = np.array([0.37,-0.34,0.045,0.35]) # these are for NACA652_env,
        self.real_obs_space_h = np.array([0.49,-0.22,0.055,0.45])
        self.TEST = 10
        self.dx = 0.1
        # dx = 0 
        self.real_dim =2 
        self.state_dim = self.real_dim


        self.transfer = transfer(dx = self.dx,dim =self.real_dim,real_obs_space_l = self.real_obs_space_l,real_obs_space_h = self.real_obs_space_h)
        # self.transfer.dim = self.real_dim
        self.log_location = self.work_location+'/log_for_2dim.txt'
        # buffer_location =  work_location+'/buffer/buffer_2dim.pkl'
        self.buffer_location =  self.work_location+'/agent0_2dim/buffer_2dim.pkl'
        self.agent0_location =  self.work_location+'/agent0_2dim'
        self.agent0_location_log = self.agent0_location + '/log_for_agent.txt'
        self.clear_txt(self.agent0_location_log)
        self.agent1_location =  self.work_location+'/agent1_2dim'
        self.converge_history_location = self.work_location+'/converge_history'
        self.env = filter_env.makeFilteredEnv(gym.make(self.ENV_NAME))
        self.env.dx = self.dx
        if abs(self.env.dx-self.dx)>0.0001:
            #which means they are not equal,
            print('MXairfoil: dx is not equal! G!')
            os.system('pause')
        self.result = np.array([]).reshape(0,4)
        # [episode,reward,x,y]
        self.episode_threshold_main = 10 
        self.episode_batch_main = 50
        self.ave_reward_threshold = 40 
    
    def set_case_Demo180(self):
        # this is for Demo180, as shown in function name.
        sys.path.append(r'C:/Users/y/Desktop/DDPGshishi/main')
        from DemoFunctions import DemoFunctions
        import Demo180_gym # just shishi if it is useful here.
        
        self.TEST = 10
        self.dx = 0.1
        # dx = 0 
        self.real_dim =18
        self.state_dim = self.real_dim


        x_yichuan1 = np.ones(self.real_dim)
        self.real_obs_space_l = x_yichuan1*(-1.0)
        self.real_obs_space_h = x_yichuan1*(1.0)
        self.transfer = transfer(dx = self.dx,dim =self.real_dim,real_obs_space_l = self.real_obs_space_l,real_obs_space_h = self.real_obs_space_h)
        self.log_location = self.work_location+'/log_for_18dim.txt'
        self.buffer_location =  self.work_location+'/agent0_18dim/buffer_18dim.pkl'
        self.agent0_location =  self.work_location+'/agent0_18dim'
        self.agent0_location_log = self.agent0_location + '/log_for_agent.txt'
        self.clear_txt(self.agent0_location_log)
        self.agent1_location =  self.work_location+'/agent1_18dim'
        self.converge_history_location = self.work_location+'/converge_history' 
        self.env = filter_env.makeFilteredEnv(gym.make(self.ENV_NAME))  
        self.result = np.array([]).reshape(0,2+self.real_dim)     
        # [episode,reward,x1,...,x18] 
        self.episode_threshold_main = 10
        self.episode_batch_main = 50
        self.ave_reward_threshold = 40 

    def set_case_Rotor67(self):
        if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
            # which means in my diannao.
            sys.path.append(r'E:\EnglishMulu\KrigingPython')
        else:
            sys.path.append(r'D:\XXHdatas\KrigingPython')
        from Surrogate_01de import Surrugate
        from Surrogate_01de import record_progress # it must be here, or class record_progress can not be found.
        import Rotor67_gym
        self.real_obs_space_h = np.array([0.07,0.14,0.21,0.03,0.06,0.09,0.48,0.15,0.16,0.15,-0.6,-0.02,-0.018,-0.12,0.26,0.7,1,1.15])
        self.real_obs_space_l = np.array([-0.07,-0.14,-0.21,-0.03,-0.06,-0.09,0.41,0.1,-0.04,-0.05,-0.67,-0.08,-0.08,-0.18,0.18,0.6,0.92,1.05])
        self.TEST = 10
        self.dx = 0.1
        # dx = 0 
        self.real_dim = 18 
        self.state_dim = self.real_dim

        self.transfer = transfer(dx = self.dx,dim =self.real_dim,real_obs_space_l = self.real_obs_space_l,real_obs_space_h = self.real_obs_space_h)
        # self.transfer.dim = self.real_dim
        self.log_location = self.work_location+'/log_for_18dim.txt'
        # buffer_location =  work_location+'/buffer/buffer_18dim.pkl'
        self.buffer_location =  self.work_location+'/agent0_18dim/buffer_18dim.pkl'
        self.agent0_location =  self.work_location+'/agent0_18dim'
        self.agent0_location_log = self.agent0_location + '/log_for_agent.txt'
        self.clear_txt(self.agent0_location_log)
        self.agent1_location =  self.work_location+'/agent1_18dim'
        self.converge_history_location = self.storage_location+'/converge_history'
        self.env = filter_env.makeFilteredEnv(gym.make(self.ENV_NAME))
        self.env.dx = self.dx
        if abs(self.env.dx-self.dx)>0.0001:
            #which means they are not equal,
            print('MXairfoil: dx is not equal! G!')
            os.system('pause')
        self.result = np.array([]).reshape(0,2+self.real_dim)
        # [episode,reward,x,y] # [episode,reward,x0,x1,x2...]
        self.episode_threshold_main = 10 
        self.episode_batch_main = 50
        self.ave_reward_threshold = 40 

        jvli_threshold=0.04
        guanghua_threshold=0.995
        self.panduan = Pjudge(dim=self.real_dim,jvli_threshold=jvli_threshold,guanghua_threshold=guanghua_threshold)
        self.jvli_Pjudge = 1
        self.guanghua_Pjudge = 0    

    def set_case_CDA2d_unconstrained(self):
        # updated from DDPGmaster82
        sys.path.append(r'C:/Users/y/Desktop/DDPGshishi/DDPG-master/CDA42-gym/KrigingPython')
        from Surrogate_01de import Surrugate
        import CDA42_gym
        self.TEST = 10
        self.dx = 0.1
        # dx = 0 
        self.real_dim = 2 
        self.state_dim = 7 
        self.real_obs_space_h = np.array([0.35,-0.22,0.55,8])
        self.real_obs_space_l = np.array([0.25,-0.38,0.35,5])
        self.transfer = transfer(dx = self.dx,dim =self.real_dim,real_obs_space_l = self.real_obs_space_l[0:self.real_dim],real_obs_space_h = self.real_obs_space_h[0:self.real_dim])

        self.log_location = self.work_location+'/log_for_2dim.txt'
        # buffer_location =  work_location+'/buffer/buffer_18dim.pkl'
        self.buffer_location =  self.work_location+'/agent0_2dim/buffer_2dim.pkl'
        self.agent0_location =  self.work_location+'/agent0_2dim'
        self.agent0_location_log = self.agent0_location + '/log_for_agent.txt'
        self.clear_txt(self.agent0_location_log)
        self.agent1_location =  self.work_location+'/agent1_2dim'
        self.converge_history_location = self.storage_location+'/converge_history'
        self.env = filter_env.makeFilteredEnv(gym.make(self.ENV_NAME))
        self.env.dx = self.dx

        if abs(self.env.dx-self.dx)>0.0001:
            #which means they are not equal,
            print('MXairfoil: dx is not equal! G!')
            os.system('pause')
        self.result = np.array([]).reshape(0,2+self.real_dim)
        # [episode,reward,x0,x1,x2...]
        self.episode_threshold_main = 10 
        self.episode_batch_main = 50
        self.ave_reward_threshold = 40 

        jvli_threshold=0.04
        guanghua_threshold=0.995
        self.panduan = Pjudge(dim=self.real_dim,jvli_threshold=jvli_threshold,guanghua_threshold=guanghua_threshold)
        self.jvli_Pjudge = 1
        self.guanghua_Pjudge = 0  

    def set_case_CDA2d_constrained(self):
        # updated from DDPGmaster82
        sys.path.append(r'C:/Users/y/Desktop/DDPGshishi/DDPG-master/CDA43-gym/KrigingPython')
        from Surrogate_01de import Surrugate
        import CDA43_gym
        self.TEST = 10
        self.dx = 0.1
        # dx = 0 
        self.real_dim = 2 
        self.state_dim = 7 
        self.real_obs_space_h = np.array([0.35,-0.22,0.55,8])
        self.real_obs_space_l = np.array([0.25,-0.38,0.35,5])
        self.transfer = transfer(dx = self.dx,dim =self.real_dim,real_obs_space_l = self.real_obs_space_l[0:self.real_dim],real_obs_space_h = self.real_obs_space_h[0:self.real_dim])

        self.log_location = self.work_location+'/log_for_2dim.txt'
        # buffer_location =  work_location+'/buffer/buffer_18dim.pkl'
        self.buffer_location =  self.work_location+'/agent0_2dim/buffer_2dim.pkl'
        self.agent0_location =  self.work_location+'/agent0_2dim'
        self.agent0_location_log = self.agent0_location + '/log_for_agent.txt'
        self.clear_txt(self.agent0_location_log)
        self.agent1_location =  self.work_location+'/agent1_2dim'
        self.converge_history_location = self.storage_location+'/converge_history'
        self.env = filter_env.makeFilteredEnv(gym.make(self.ENV_NAME))
        self.env.dx = self.dx

        if abs(self.env.dx-self.dx)>0.0001:
            #which means they are not equal,
            print('MXairfoil: dx is not equal! G!')
            os.system('pause')
        self.result = np.array([]).reshape(0,2+self.real_dim)
        # [episode,reward,x0,x1,x2...]
        self.episode_threshold_main = 10 
        self.episode_batch_main = 50
        self.ave_reward_threshold = 40 

        jvli_threshold=0.04
        guanghua_threshold=0.995
        self.panduan = Pjudge(dim=self.real_dim,jvli_threshold=jvli_threshold,guanghua_threshold=guanghua_threshold)
        self.jvli_Pjudge = 1
        self.guanghua_Pjudge = 0  


    def set_case_CDA4d(self):
        sys.path.append(r'C:/Users/y/Desktop/DDPGshishi/DDPG-master/CDA44-gym/KrigingPython')
        from Surrogate_01de import Surrugate
        import CDA44_gym        
        self.TEST = 10
        self.dx = 0.1
        # dx = 0 
        self.real_dim = 4 
        self.state_dim = 7 
        self.real_obs_space_h = np.array([0.35,-0.22,0.55,8])
        self.real_obs_space_l = np.array([0.25,-0.38,0.35,5])
        self.transfer = transfer(dx = self.dx,dim =self.real_dim,real_obs_space_l = self.real_obs_space_l[0:self.real_dim],real_obs_space_h = self.real_obs_space_h[0:self.real_dim])

        self.log_location = self.work_location+'/log_for_4dim.txt'
        # buffer_location =  work_location+'/buffer/buffer_18dim.pkl'
        self.buffer_location =  self.work_location+'/agent0_4dim/buffer_4dim.pkl'
        self.agent0_location =  self.work_location+'/agent0_4dim'
        self.agent0_location_log = self.agent0_location + '/log_for_agent.txt'
        self.clear_txt(self.agent0_location_log)
        self.agent1_location =  self.work_location+'/agent1_4dim'
        self.converge_history_location = self.storage_location+'/converge_history'
        self.env = filter_env.makeFilteredEnv(gym.make(self.ENV_NAME))
        self.env.dx = self.dx

        if abs(self.env.dx-self.dx)>0.0001:
            #which means they are not equal,
            print('MXairfoil: dx is not equal! G!')
            os.system('pause')
        self.result = np.array([]).reshape(0,2+self.real_dim)
        # [episode,reward,x0,x1,x2...]
        self.episode_threshold_main = 10 
        self.episode_batch_main = 50
        self.ave_reward_threshold = 40 

        jvli_threshold=0.04
        guanghua_threshold=0.995
        self.panduan = Pjudge(dim=self.real_dim,jvli_threshold=jvli_threshold,guanghua_threshold=guanghua_threshold)
        self.jvli_Pjudge = 1
        self.guanghua_Pjudge = 0 

        self.chicun = [400,300]


    def main(self,**kargs):
        if not(self.init_agents_flag):
            self.init_agents()
        
        # env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
        env=self.env
        # agent0 = DDPG(env)
        buffer = self.buffer
        self.tiaochu = 0 

        agent0_location= self.agent0_location
        buffer_location = self.buffer_location
        EPISODES = self.EPISODES

        if 'agent1' in  kargs:
            agent1 = kargs['agent1']
        else:
            agent1 = self.agent1

        if 'agent0' in  kargs:
            agent0 = kargs['agent0']
            # env = agent0.environment
            # buffer =  agent0.replay_buffer
        else:
            agent0 = self.agent0

        if len(self.result) == 0 :
            self.result = np.append(self.result,np.zeros((1,2+self.real_dim)),axis=0)
            # this is for compatibility

        tiaochu=False
        flag0=0
        flag1=0
        saved0 = 0 
        saved1 = 0 
        while buffer.count() < self.REPLAY_START_SIZE: 
            bili = (buffer.count()/self.REPLAY_START_SIZE)*100
            print('MXairfoil: feeding buffer: '+str(bili)+'% ...')
            # feed_buffer_mul(agent0,agent1,ENV_NAME,1)
            self.feed_buffer_single(agent0,env)

        buffer.save_buffer(buffer_location)
        print('MXairfoil: finish feeding the buffer. Now, start train')

        if self.zuobi:
            raw_state_new = np.array([-0.83,0.83])
            state,reward,done,_ = env.set_state(raw_state_new)
            raw_state_performance_new = env.get_performance()*1.0
            env.set_raw_state(raw_state_new)
            agent0.record_raw_state(raw_state_new[0:self.real_dim],episode=0,performance = raw_state_performance_new)
            env.reset_random()

        # then start the iteration.
        for episode in range(EPISODES):
            # training parallel
            # feed_buffer_mul(agent0,agent1,ENV_NAME,N_threads)
            if (episode % 10 == 9):
                print('MXairfoil: episode = '+str(episode))
            if self.kaiguan()>0:
                print('MXairfoil: forced jump out of a loop')
                break
            
            self.feed_buffer_single(agent0,env)

            # Testing:
            if episode < self.episode_min:
                episode_batch = self.episode_batch_main * 2
            else:
                episode_batch = self.episode_batch_main 

            if (episode % episode_batch == episode_batch-1) or (episode<=self.episode_threshold_main) :
                # agent0.save_agent(agent0_location,episode) # this is for debug, to see untrainned agent.

                flag0,ave_reward0 = self.average_test(episode,env,agent0,flag0)
                # flag1 = average_test(episode,env,agent1,flag1)
                flag1 = 1 
                self.ave_reward0 = ave_reward0

                # agent0.record_ave_reward(episode,ave_reward0)

                self.tiaochu = flag0 & flag1 # all are done, then tiaochu

                if flag0 == 0:
                    #which means it is still not good enough
                    # agent0.save_agent(agent0_location,episode)
                    print('MXairfoil: anget0 is not good enough')
                    if ave_reward0 > agent0.ave_reward_save0[:,1].max():
                        print('    but is better than last ones')
                        agent0.save_agent(agent0_location,episode)
                elif saved0==0:
                    # agent0.save_agent(agent0_location,episode)
                    print('MXairfoil: anget0 has not been done. no need.')
                    saved0 =1 

                agent0.record_ave_reward(episode,ave_reward0)
                
            if self.tiaochu :
                print('MXairfoil: jump out of a loop')
                break
        

        
        if episode > EPISODES-3 :
            agent0.save_agent(agent0_location,episode)
            print('MXairfoil: EPISODES has been exhausted.')
        # env.close()
        self.record_history(agent0.ave_reward_save0,agent0.raw_state_save)
        # self.huatu_for_main(agent0)
        try:
            self.huatu_for_main(agent0)
        except:
            print('MXairfoil: fail to huatu for main ')
        return agent0 , agent1

    def feed_buffer_single(self,agent0,env):
        # this is for surrogate model env. we don't need threading parallel anymore 
        # state = env.reset()*1
        # state = env.reset_random2()*1
        if self.ENV_NAME == 'NACA652_env-v0':
            if agent0.raw_state_performance[-1][-1]<0.7:
                state = env.reset_random2()*1.0 # this is for total random
            else:
                suiji = random.randint(0,10)
                if suiji < 4:
                    state = env.reset_random()*1.0
                else:
                    state = env.reset_random2()*1.0
        elif self.ENV_NAME == 'Demo180_env-v0':
            state = env.reset_random2()*1.0 # this is for total random
        elif self.ENV_NAME == 'Rotor67_env-v0':
            state = env.reset()*1.0 
        elif self.ENV_NAME == 'CDA42_env-v0':
            chicun = agent0.raw_state_save.shape
            if agent0.raw_state_save[chicun[0] - 1][chicun[1] - 1]<0.7:
                state = env.reset_random2()*1
            else:
                state = env.reset_random()*1   
        elif self.ENV_NAME == 'CDA43_env-v0': 
            state = env.reset_random()*1
        else:
            state = env.reset()*1.0
            
        try:
            env._has_reset = True
        except:
            pass

        for step in range(self.N_steps):
            action = agent0.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            # agent0.perceive(state,action,next_state[4+agent0.kind],next_state,done)
            agent0.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break

    def average_test(self,episode,env,agent,flag):
        real_dim = self.real_dim
        TEST = self.TEST
        buffer_location = self.buffer_location
        #simplify the code by defining another function.
        # flag = 0 #0 for continue, 1 for good enough.
        if flag == 1 :
            return flag
            #which means this one is  good enough
        kind = agent.kind
        total_reward = 0
        raw_state_new = agent.raw_state_save[-1]*1.0
        raw_state_performance_new =agent.raw_state_performance[-1]*1.0
        for i in range(TEST):
            # state = env.reset_random()*1
            # if agent.raw_state_save[chicun[0] - 1][chicun[1] - 1]<0.7:
            if self.feed_back_enable :
                fenjie = 5 # high-order feedback on, so more test using limited random.
            else:
                fenjie = 9
                # change the architecture into oop, I'm so jizhi!

            if i<fenjie:
                state = env.reset_random2()*1 # totally random.
                # if it is not good, random search.
            else :
                state = env.reset_random()*1 # limited random.
            # for j in range(env.spec.timestep_limit): 
            for j in range(self.N_steps):
                #env.render()
                action = agent.action(state) # direct action for test
                state,reward,done,_ = env.step(action)
                total_reward += reward
                performance = env.get_performance()*1.0
                # performance is newer here, raw_state_performance_new is to be update.
                # try: 
                #     raw_state_performance_new =raw_state_performance_new[-1]
                # except:
                #     raw_state_performance_new = raw_state_performance_new 
                if raw_state_performance_new[-1] < performance[-1]: # when new reward is higher, update. # this is for demo functions. 
                # if raw_state_performance_new[0] > performance[0]: # when new omega(in performance) is lower, update. # this is for CDA environments
                    raw_state_new = state * 1.0
                    raw_state_performance_new = performance* 1.0
                    # get the real 'best' one for uploading.
                
                if done:
                    break
        ave_reward = total_reward/TEST

        strbuffer = '\n\nepisode: '+str(episode)+'\nagent id:'+str(kind)+'\nEvaluation Average Reward:'+str(ave_reward) + '\nbuffer size: ' + str(agent.replay_buffer.count())+'\nartificial tip used number: '+ str(env.N_artificial_tip)
        self.jilu(strbuffer)
        self.result[-1][0] = episode* 1.0
        self.result[-1][1] = ave_reward* 1.0

        if (agent.replay_buffer.count() >agent.REPLAY_START_SIZE):
            agent.save_buffer(buffer_location)
        
        
        if ave_reward>self.ave_reward_threshold  or (episode<=self.episode_threshold_main) : 
            yuzhi = 0   # yuzhi = 0 for nothing happen 
            # yuzhi = 0.0003 # yizhi != 0 for loosenning the limite and exploring more
            # chicun = agent.raw_state_save.shape
            if (episode ==114514)&(self.ENV_NAME == 'NACA652_env-v0') :
                # this is for zuobi
                raw_state_new[0] = -0.33333333  
                raw_state_new[1] = 0.83333333
                raw_state_new[-1] = 1.0

            if (episode>self.episode_min/5) and (self.ENV_NAME == 'Rotor67_env-v0'):
                flag_Pjudge = self.average_test_Pjudge(agent=agent,episode=episode)
                flag = flag and flag_Pjudge 
                # if Pjudge test success, then stop immediately 
            # try:
            #     zhongjie = raw_state_performance_new[-1]
            # except:
            #     zhongjie = raw_state_performance_new
            # try:
            #     zhongjie2 = agent.raw_state_performance[-1][-1] 
            # except:
            #     zhongjie2 = agent.raw_state_performance[-1] 
            zhongjie = raw_state_performance_new[-1]
            zhongjie2 = agent.raw_state_performance[-1][-1] 
            if ((zhongjie > (zhongjie2-yuzhi))or (episode==10)) and self.feed_back_enable :  # reward itself, the bigger the better.
            # if ((raw_state_performance_new[0] < (agent.raw_state_performance[-1][0]-yuzhi))or (episode==10)) and self.feed_back_enable : # in CDA env, update by omega, lower is better.    
                # update only when the newer is better
                env.set_raw_state(raw_state_new)
                agent.record_raw_state(raw_state_new[0:self.real_dim],episode=episode,performance = raw_state_performance_new)
                rizhi = '\nMXairfoil: raw stata updated. \nbefore:'+str(agent.raw_state_save[-2]) + '    performance:'+str(agent.raw_state_performance[-2])+'\nafter'+str(raw_state_new) + '    performance:'+str(raw_state_performance_new)+ '\nartificial tip used number: '+ str(env.N_artificial_tip)
                self.jilu(rizhi)
            else:
                if self.feed_back_enable == False:
                    print('MXairfoil: high order feedback disabled')



        if (ave_reward  >self.ave_reward_threshold) and (episode>self.episode_min) :
            # this threshold would be changed dynamicly
            print('mytest and save agent,average reward=',ave_reward)
            flag = 1
            print('MXairfoil: it looks there are something relatively good, jump out of a loop')
            # self.result[-1][0] = episode
            # self.result[-1][1] = ave_reward

        return flag,ave_reward

    def average_test_Pjudge(self,agent,episode):
        # self.jvli_Pjudge = 1
        # self.guanghua_Pjudge = 1
        lujing_list = self.saved_agent_test(agent0=agent,result_kind='list')
        flag,jvli,guanghua,strbuffer_Pjudge = self.panduan.Pjudge_test(lujing_list,model = 'value')
        
        if jvli<self.jvli_Pjudge:
            strbuffer_Pjudge  = strbuffer_Pjudge + ' \n jvli improved, save'
        agent.save_agent(self.agent0_location,episode)        
        self.jvli_Pjudge = jvli*1.0
        self.guanghua_Pjudge = guanghua*1.0

        self.jilu(strbuffer_Pjudge)

        return flag

    def kaiguan(self):
        # this is a simple on-off. if the location is no exists, then nothing would happen, if I buid it, then main() should end.
        location = self.kaiguan_location
        if os.path.exists(location) :
            zhi = 1 # for end the program
            rizhi = 'MXairfoil: end the program by hand, jieguo = '+str(self.result)
            self.jilu(rizhi)
        else :
            zhi = 0 # for nothing happen
        return zhi

    def huatu_for_main(self,agent0):
        agent0_location = self.agent0_location
        tu = huatu(agent0.ave_reward_save0)
        tu.set_location(agent0_location)
        tu.huatu2D('episode','average reward','Reward-Episode relation')
        tu.save_all()

        # draw a picture for raw_state.
        changdu = len(agent0.raw_state_save)
        if self.ENV_NAME == 'Demo180_env-v0':
            # it seems additional work shall be done.
            # agent0.raw_state_performance[-1]*1.0
            shuru = np.append(agent0.raw_state_episode[:,0].reshape(changdu,1),agent0.raw_state_performance[:,-1].reshape(changdu,1),axis=1)
            # shuru = np.append(agent0.raw_state_episode[:,0].reshape(changdu,1),agent0.raw_state_performance[:,0].reshape(changdu,1),axis=1)
        else:
            shuru = np.append(agent0.raw_state_episode[:,0].reshape(changdu,1),agent0.raw_state_save[:,-1].reshape(changdu,1),axis=1)
        tu2 = huatu(shuru)
        tu2.set_location(agent0_location)
        tu2.huatu2D('episode','Optimization reward','Optimization-Episode relation')
        tu2.save_all()

        # merge them
        weizhi = agent0_location
        location1 = weizhi+'/Optimization-Episode relation'
        location2 = weizhi+'/Reward-Episode relation'
        tu.set_location(weizhi)
        tu.load_data_mul(location1,location2)
        tu.huatu2D_mul2('episode','average reward','Converge History','optimization detected','average reward')
        tu.save_all()

    def jilu(self,strBuffer):
        shijian = time.strftime("%Y-%m-%d-%H:%M", time.localtime()) 
        wenjianming = self.log_location
        rizhi = open(wenjianming,'a')
        rizhi.write(strBuffer)
        rizhi.write('\n'+shijian+'\n')
        rizhi.close()
        print(strBuffer)
        try:
            wenjianming = self.agent0_location_log
            rizhi = open(wenjianming,'a')
            rizhi.write(strBuffer)
            rizhi.write('\n'+shijian+'\n')
            rizhi.close()
        except:
            pass 
        return
    
    def saved_agent_test(self,**kargs):
        # this is trying to load a saved agent, and get something looks like optimilzed result.
        env = self.env
        if 'location' in kargs:
            agent0_location = kargs['location']
        else:
            agent0_location = self.agent0_location
        buffer_location = self.buffer_location
        real_dim = self.real_dim
        TEST = self.TEST
        if 'agent0' in kargs:
            print('MXairfoil: agent0 inputed')
            agent0 = kargs['agent0']
        else:
            buffer = ReplayBuffer(self.REPLAY_BUFFER_SIZE)
            agent1 = self.agent1
            agent0 = self.agent0
            agent0.load_agent(agent0_location)
            agent0.replay_buffer.load_buffer(buffer_location)
        
        if 'constraints' in kargs:
            constraints = kargs['constraints']
            try:
                wenjianjia_constraints = '/'+ str(constraints)
                os.mkdir(agent0_location + wenjianjia_constraints)
            except:
                print('MXairfoil:fail to make a dir for saved_agent_test')
        else:
           wenjianjia_constraints = ''

        if 'result_kind' in kargs:
            result_kind = kargs['result_kind'] # 'list' to return lujing_list
        else:
            result_kind = 'location'

        test_step = self.N_steps
        global_optimization = agent0.raw_state_save[len(agent0.raw_state_save)-1]
        # if len(global_optimization) != self.real_dim:
        #     gai = np.zeros(self.real_dim,dtype=float)
        #     gai[0:len(global_optimization)] = global_optimization
        #     global_optimization = gai
        real_go = self.transfer.normal_to_real(global_optimization)
        real_go[2:4] = global_optimization[2:4]
        rizhi = 'MXairfoil: raw_state = ' + str(global_optimization)+'\nin real space:'+ str(real_go) 
        self.jilu(rizhi)
        kind = agent0.kind
        total_reward = 0
        states_array = np.array([]).reshape(0,self.state_dim) # this reshape is for appending 
        lujing = np.array([]).reshape(0,self.state_dim)
        performance = np.array([]).reshape(0,self.env.performance_dim)

        lujing_list = [] 

        for i in range(TEST):
            if self.ENV_NAME == 'Demo180_env-v0':
                state = env.reset_random2()
            else:
                if i==0:
                    state = env.reset()*1
                elif i ==1:
                    #get one for real state. # there are some hidden danger here, reset_original is directly called without filter_env.
                    state = env.reset_original()*1
                else:
                    state = env.reset_random2()*1

            # set constant constraints rather than dynamic 
            # this would overwrite the constraints in env
            # state = env.set_constraints(constraints)

            for j in range(test_step):
                state2 = state*1
                lujing = np.append(lujing,self.transfer.normal_to_surrogate(state2).reshape(1,len(state2)),axis=0)
                performance = np.append(performance,env.get_performance().reshape(1,env.performance_dim),axis=0)

                action = agent0.action(state) # direct action for test
                state,reward,done,_ = env.step(action)
                total_reward += reward
                
                if done:
                    break

            # save the lujing for further use.
            wenjianming_lujing = agent0_location + wenjianjia_constraints+'/lujing' + str(i) + '.pkl'
            pickle.dump(lujing,open(wenjianming_lujing,'wb'))
            # record the lujing for Pjudge.
            lujing_list.append(lujing)
            # save the performance for further use.
            wenjianming_performance = agent0_location + wenjianjia_constraints + '/performance' + str(i) + '.pkl'
            pickle.dump(performance,open(wenjianming_performance,'wb'))

            lujing = np.array([]).reshape(0,self.state_dim) # reset this after saved
            performance = np.array([]).reshape(0,self.env.performance_dim) # reset this after saved

            states_array = np.append(states_array,state.reshape(1,self.state_dim),axis=0)
            
        ave_reward = total_reward/TEST
        state_optimized = states_array.sum(0)/len(states_array)
        # rizhi = '\n\nMXairfoil: saved_agent_test done. Using this traind agent, optimized state is: ' + str(state_optimized)
        rizhi = '\n\nMXairfoil: saved_agent_test done.'
        self.jilu(rizhi)
        print('MXairfoil: maybe...')
        if result_kind == 'location':
            return agent0_location + wenjianjia_constraints
        elif result_kind == 'list':
            return lujing_list
        else:
            return 0 
    
    def huatu_post(self,**kargs):   
        if 'location' in kargs:
            loaction = kargs['location']
            # copy backgroud into it.
            wenjianing_X1 = self.agent0_location + '/visual2DX1.pkl'
            wenjianing_X2 = self.agent0_location + '/visual2DX2.pkl'
            wenjianing_Y1 = self.agent0_location + '/visual2DY1.pkl'
            wenjianing_Y2 = self.agent0_location + '/visual2DY2.pkl'
            wenjianing_Y3 = self.agent0_location + '/visual2DY3.pkl'
            try:
                shutil.copy(wenjianing_X1,loaction)
                shutil.copy(wenjianing_X2,loaction)
                shutil.copy(wenjianing_Y1,loaction)
                shutil.copy(wenjianing_Y2,loaction)
                shutil.copy(wenjianing_Y3,loaction)
            except:
                print('MXairfoil: fail to get backgroud data, G!')
        else:
            loaction = self.agent0_location


        try:
            self.agent0.load_agent(self.agent0_location)
            # self.result = np.array([0.0,0.0,0.0,0.0]).reshape(1,4)
            print('MXairfoil: agent loaded in huatu_post')
        except:
            print('MXairfoil: there are agent already')
        if 'only_data' in kargs:
            only_data = kargs['only_data']
        else:
            only_data = False

        tu = huatu(0)
        # huatu for agent 2d.
        try:
            print('MXairfoil: huatu in '+loaction)
            tu.set_location(loaction)
            tu.visual_2D(1,0) 
            # tu.visual_2D(2,0)
            for i in range(10):
                tu.visual_2D(4,i,only_data=only_data) 
                # tu.visual_2D(4,i) 
        except:
            print('MXairfoil: huatu fail in '+loaction)
        end_point = np.mean(tu.end_point,axis=0)
        self.result[-1][2:2+self.real_dim] = end_point 
        self.result[-1][2] = end_point[0]
        self.result[-1][3] = end_point[1]
        
        if self.ENV_NAME == 'Demo434_env-v0':
            lilunzhi = np.array([0.3,1.0])
        else:
            lilunzhi = self.agent0.raw_state_save[-1]
        lilunzhi = self.transfer.normal_to_surrogate(lilunzhi)
        jvli = np.sum( (end_point - lilunzhi)**2) **(0.5)
        return jvli

    def auto_run(self,**kargs):
        jvli = 114514
        rizhi = 'MXairfoil: auto_run start, En Taro XXH!'
        agent0_location = self.agent0_location
        if 'auto_steps' in kargs:
            auto_steps=kargs['auto_steps']
        else:
            auto_steps = 10
        # while jvli>0.05:
        for i in range(auto_steps):
            if self.kaiguan()>0:
                print('MXairfoil: forced jump out of auto_run')
                break
            self.result = np.append(self.result,np.zeros((1,2+self.real_dim)),axis=0)
            agent0 , agent1 = self.main()
            mulu = self.saved_agent_test(agent0=agent0)
            jvli_new = self.huatu_post(location=mulu,only_data=True)
            rizhi = 'MXairfoil: this is loop'+str(i)
            if   jvli_new<jvli + 114514:
                rizhi = '\n jvli = '+ str(jvli_new)
                try:
                    agent0_location_record = agent0_location+str(jvli_new)
                    os.mkdir(agent0_location_record)
                    agent0.save_agent(agent0_location_record,1)
                    mulu = self.saved_agent_test(agent0=agent0,location = agent0_location_record)
                    self.huatu_post(location=agent0_location_record,only_data=False)
                    rizhi = rizhi + '\ntryed to save this agent.'
                except:
                    rizhi = rizhi + '\nfail to save record agent.'
                jvli =  jvli_new
            else:
                rizhi = rizhi+'\nno need to save this agent.'
            agent0.reset_agent(location =agent0_location) 
            self.agent0.reset_agent()  
            rizhi = rizhi +'\nresult = ' +str(self.result) 
            self.jilu(rizhi)
            self.auto_record()
        agent0.save_agent(agent0_location_record,1)         

    def auto_record(self,total_time_cost = 0 ,**kargs):
        # auto record the configuration in a single txt.
        print('MXairfoil: Auto record the configuration and result in txt')
        shijian = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
        if 'location' in kargs:
            wenjianming = kargs['location'] + '\configuration.txt'
        else:
            wenjianming = self.agent0_location + '\configuration.txt'
        wenjian = open(wenjianming,'w')
        wenjian.write('MXairfoil:configuration recorded. En Taro XXH!\n')
        wenjian.write('\nreal_dim='+str(self.real_dim))
        wenjian.write('\nENV_NAME='+str(self.ENV_NAME))
        wenjian.write('\nEPISODES='+str(self.EPISODES))
        wenjian.write('\nepisode_min='+str(self.episode_min))
        wenjian.write('\nnetworks:'+str(self.agent0.get_config()))
        wenjian.write('\nfinished at:'+shijian)
        wenjian.write('\ntotal_time_cost:'+str(total_time_cost))
        try:
            neirong = self.jishiqi.get_result(model='str')
        except:
            neirong = 'wrong'
        wenjian.write('\njishiqi: ' + neirong)
        wenjian.close()
    
    def record_history(self,ave_reward_save0,opt_reward):
        # this is to record the converge history of this round of trainning.
        # load, append, then saved
        wenjianming = self.converge_history_location + '/converge_history.pkl'
        if os.path.exists(self.converge_history_location):
            pass
        else:
            os.mkdir(self.converge_history_location)
        try:
            self.converge_history = pickle.load(open(wenjianming,'rb'))
        except:
            self.converge_history = [] 

        # agent0.ave_reward_save0
        self.converge_history.append(ave_reward_save0)
        N_saved = len(self.converge_history) 
        print('MXairfoil: '+str(N_saved)+' ave trainnings have been recorded')

        # then save back.
        pickle.dump(self.converge_history,open(wenjianming,'wb'))

        # then another one,
        wenjianming2 = self.converge_history_location + '/converge_history_opt.pkl'
        try:
            self.converge_history_opt = pickle.load(open(wenjianming,'rb'))
        except:
            self.converge_history_opt = []     
        self.converge_history_opt.append(opt_reward)    
        N_saved2 = len(self.converge_history_opt) 
        print('MXairfoil: '+str(N_saved2)+' opt trainnings have been recorded')
        # then save back.
        pickle.dump(self.converge_history,open(wenjianming2,'wb'))

    def clear_txt(self,name):
        # just clear the txt.
        try:
            file = open(name, 'w').close()
        except:
            # if on such file,creat one.
            file = open(name, 'w').close()

    def auto_run2(self,jvli_threshold=0.04,guanghua_threshold=0.995,episode_batch_main=50,**kargs):
        # this is for demo180 first. just train and record some agents.
        rizhi = 'MXairfoil: auto_run start, En Taro XXH!'
        agent0_location = self.agent0_location
        self.episode_batch_main = episode_batch_main 
        # some setting items.
        if 'auto_steps' in kargs:
            auto_steps=kargs['auto_steps']
        else:
            auto_steps = 10
        
        if 'model' in kargs:
            model = kargs['model']
        else:
            model = 'simple'

        if model == 'simple':
            flag_filtrate = False
        elif model == 'filtrate':
            flag_filtrate = True

        n=0
        panduan = Pjudge(dim=self.real_dim,jvli_threshold=jvli_threshold,guanghua_threshold=guanghua_threshold)
        while n<auto_steps:
            total_time_start = time.time()
            agent0 , agent1 = self.main()
            total_time_end = time.time()
            total_time_cost = total_time_end - total_time_start
            self.auto_record(total_time_cost=total_time_cost)

            # then save to storage.
            self.check_exist_agent(agent0,panduan,flag_filtrate=flag_filtrate,index=n)
            
            if self.ave_reward0 > self.ave_reward_threshold-10:
                pass 
                # self.check_exist_agent(agent0,panduan,flag_filtrate=flag_filtrate,index=n)
                '''
                lujing_list = self.saved_agent_test(agent0=agent0,result_kind='list')
                flag_panju,strbuffer = panduan.Pjudge_test(lujing_list,model = 'str_on')
                strbuffer = strbuffer + '\ntime cost for one round: ' + str(total_time_cost)
                self.jilu(strbuffer)
                if flag_panju or (not(flag_filtrate))  :
                    # which means successfully pass the Pjudge
                    self.save_agents_result(self.agent0_location,index=n)
                    n=n+1
                    print('MXairfoil: one of the agent has been recorded, n='+str(n))
                '''
            # then reset
            agent0.reset_agent(location =self.agent0_location) 
            self.agent0.reset_agent(location =self.agent0_location)  
            self.clear_agents()
        # self.huatu_for_history()

    def save_agents_result(self,location_temp,**kargs):
        # this is to copy cases into HDD
        index = kargs['index']
        location_target = self.storage_location + '/agent0indedx'+str(index)
        try:
            shutil.copytree(location_temp,location_target)
        except:
            print('MXairfoil: warning! repetitive case?')
            # i = 0 
            # while (i<30) and (os.path.exists(location_target2)):
            while (index<30) :
                location_target2 = self.storage_location + '/agent0indedx'+str(index)
                if (os.path.exists(location_target2)):
                    index=index+1
                else:
                    try:
                        shutil.copytree(location_temp,location_target2)
                        print('MXairfoil: repetitive case renamed:' + location_target2)
                        break
                    except:
                        i=i+1

    def clear_agents(self,**kargs):
        # this is to clear work_location, which is **/agents.
        clear_agent_location = self.agent0_location
        try:
            shutil.rmtree(clear_agent_location)
            print('MXairfoil: clear_agent_location successfully')
            os.mkdir(clear_agent_location)
        except:
            print('MXairfoil: clear_agent_location fail')      
        
        try:
            self.clear_txt(self.log_location)
            print('MXairfoil: clear_log_location successfully')
        except:
            print('MXairfoil: clear_log_location fail')      

        try:
            os.remove(self.kaiguan_location)
            print('MXairfoil: clear_kaiguan_location successfully')
        except:
            print('MXairfoil: clear_kaiguan_location fail') 

        if 'clear_storage' in kargs:
            if kargs['clear_storage'] == True:
                try:
                    shutil.rmtree(self.storage_location)
                    print('MXairfoil: clear_storage_location successfully')
                    os.mkdir(self.storage_location)
                except:
                    print('MXairfoil: clear_storage_location fail') 

        if 'clear_converge_history' in kargs:
            if kargs['clear_converge_history'] == True:
                try:
                    shutil.rmtree(self.converge_history_location)
                    print('MXairfoil: clear_converge_history_location successfully')
                    os.mkdir(self.converge_history_location)
                except:
                    print('MXairfoil: clear_converge_history_location fail')  

    def load_exist_agent(self):
        # load existing agent
        self.buffer_reuse=True
        self.agent_reuse = True
        self.init_agents_flag= False
        self.init_agents()

    def check_exist_agent(self,agent0,panduan,flag_filtrate=False,index=0):
        lujing_list = self.saved_agent_test(agent0=agent0,result_kind='list')
        flag_panju,strbuffer = panduan.Pjudge_test(lujing_list,model = 'str_on')
        self.jilu(strbuffer)
        if flag_panju or (not(flag_filtrate))  :
            # which means successfully pass the Pjudge
            self.save_agents_result(self.agent0_location,index=index)
            print('MXairfoil: one of the agent has been recorded, n='+str(index))
    
    def recycle_exist_agent(self,jvli_threshold=0.04,guanghua_threshold=0.995,flag_filtrate = False):
        self.load_exist_agent()
        panduan = Pjudge(dim=self.real_dim,jvli_threshold=jvli_threshold,guanghua_threshold=guanghua_threshold) 
        self.check_exist_agent(agent0=self.agent0,panduan=panduan,flag_filtrate=flag_filtrate)
        print('MXairfoil: recycle_exist_agent done.')
if __name__ == '__main__':
    total_time_start = time.time()
    flag =2
    # 0 for huatu, 1 for run.
    # 2 for new democase of 18 dim.
    # -999 for clear.
    if flag==0:
        shishi = auto_jisuan(dx=0.1,feed_back_enable=True,ENV_NAME = 'NACA652_env-v0',ave_reward_threshold=92,buffer_reuse=False,EPISODES=700,agent_reuse = False,zuobi=False)
        # constraints = np.array([[-1.0,-1.0]])
        mulu = shishi.saved_agent_test()
        shishi.huatu_post(location=shishi.agent0_location)
    elif flag == 1 :
        shishi = auto_jisuan(dx=0.1,feed_back_enable=True,ENV_NAME = 'NACA652_env-v0',ave_reward_threshold=92,buffer_reuse=False,EPISODES=700,agent_reuse = False,zuobi=False)
        shishi.auto_run(auto_steps=30)
    elif flag == 2 :
        shishi = auto_jisuan(dx=0.1,feed_back_enable=True,ENV_NAME = 'Demo180_env-v0',ave_reward_threshold=57,buffer_reuse=True,EPISODES=7000,agent_reuse = False,zuobi=False)
        # shishi.main()
        shishi.auto_run2(auto_steps=30,model='filtrate')
    elif flag == -999 :
        shishi = auto_jisuan(dx=0.1,feed_back_enable=True,ENV_NAME = 'Demo180_env-v0',ave_reward_threshold=50,buffer_reuse=True,EPISODES=1000,agent_reuse = False,zuobi=False)
        shishi.clear_agents(clear_storage=True)        
    total_time_end = time.time()
    total_time_cost = total_time_end - total_time_start
    print('MXairfoil: total time cost ='+str(total_time_cost))