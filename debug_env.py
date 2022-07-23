# this is for showing the inderactive window.

from tkinter import N
import numpy as np 
import time 
import os 

from main_auto import auto_jisuan
import psutil
import pickle
import sys

if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
    # which means in my diannao
    WEIZHI = r'E:\EnglishMulu\KrigingPython'
    sys.path.append(WEIZHI)
else :
    # which means in server
    WEIZHI =r'D:/XXHcode/DDPGshishi/DDPG-master/Rotor67-gym/KrigingPython'
    sys.path.append(WEIZHI)
from Surrogate_01de import Surrugate
from Surrogate_01de import record_progress

import gym
class debug_env(auto_jisuan):
    def __init__(self) :
        # import Demo180_gym # just shishi if it is useful here.
        # gym.make('Demo180_env-v0')
        # import Rotor67_gym
        # gym.make('Rotor67_env-v0')
        super().__init__(ENV_NAME = 'Rotor67_env-v0',dx=0.1)
        print('MXairfoil: confirm env.a1=1,env.a2=0 and reset into real space, before calculating weight.')
        pass

    def standardization_performance(self,N_points = 114514):
        # calculate several steps to get perfromance_w and performance_b 
        performance_all = np.zeros((N_points,self.env.performance_dim))
        weizhi = self.work_location + '/performance_all.pkl'
        if os.path.exists(weizhi):
            return
        total_time_start = time.time()
        for i in range(N_points):
            # caonima
            performance_all[i] = self.get_random_performance(self.env)
            print('MXairfoil: ' + str(i) + ' points done.')
        
        total_time_end = time.time()
        total_time_cost = total_time_end - total_time_start
        print('MXairfoil: total time cost ='+str(total_time_cost) + ', for ' + str(N_points) + ' steps')     
        
        # self.X = pickle.load(open(location_X,'rb'))
        pickle.dump(performance_all,open(weizhi,'wb'))

    def get_random_performance(self,env):
        env.reset()
        performance = env.get_performance()
        return performance

    def get_random_reward(self,env):
        env.reset()
        state, reward, done, _ = env.step(np.zeros(env.real_dim))
        return reward 

    def calculate_weight_performance(self):
        # map the performance into [0,1]
        weizhi = self.work_location + '/performance_all.pkl'
        self.performance_all = pickle.load(open(weizhi,'rb'))

        # if self.performance_all:
        #     self.performance_all = pickle.load(open(weizhi,'rb'))
        performance_min = np.min(self.performance_all,axis=0)
        performance_max = np.max(self.performance_all,axis=0)
        performance_cha = performance_max - performance_min
        
        w = 1.0 / performance_cha
        b = 0 - w*performance_min
        performance_all_normal = w*self.performance_all + b
        check_1 = np.max(np.max(performance_all_normal))
        check_0 = np.min(np.min(performance_all_normal))
        if abs(check_1-1) + abs(check_0-0) < 0.00000001:
            print('MXairfoil: successfully get weights. \n w='+ str(w) +'\n b=' + str(b))

    def standardization_reward(self,N_points=114514,reuse = True):
        # calculate several steps to get parameters for r.
        # calculate several steps to get perfromance_w and performance_b 
        reward_all = np.zeros((N_points,1))
        weizhi = self.work_location + '/reward_all.pkl'
        if os.path.exists(weizhi) and reuse:
            return
        total_time_start = time.time()
        for i in range(N_points):
            # caonima
            reward_all[i] = self.get_random_reward(self.env)
            print('MXairfoil: ' + str(i) + ' points done.')
        
        total_time_end = time.time()
        total_time_cost = total_time_end - total_time_start
        print('MXairfoil: total time cost ='+str(total_time_cost) + ', for ' + str(N_points) + ' steps')     
        
        # self.X = pickle.load(open(location_X,'rb'))
        pickle.dump(reward_all,open(weizhi,'wb'))        

    def calculate_weight_reward(self):
        # map the performance into [0,1]
        weizhi = self.work_location + '/reward_all.pkl'
        self.reward_all = pickle.load(open(weizhi,'rb'))

        # performance_min = np.min(self.reward_all,axis=0)
        # performance_max = np.max(self.reward_all,axis=0)
        pingjun = np.mean(self.reward_all)
        performance_min = 0
        performance_max = 0 
        N_ave = 3
        for i in range(N_ave):
            # more point, to avoid pinnacle 
            index_min  = np.argmin(self.reward_all,axis=0)
            index_max  = np.argmax(self.reward_all,axis=0)
            performance_min = performance_min + self.reward_all[index_min]
            performance_max = performance_max + self.reward_all[index_max]
            self.reward_all[index_min] = pingjun
            self.reward_all[index_max] = pingjun 

        performance_min = performance_min / N_ave
        performance_max = performance_max / N_ave
        performance_cha = performance_max - performance_min
        
        w = 1.0 / performance_cha
        b = 0 - w*performance_min
        reward_all_normal = w*self.reward_all + b
        check_1 = np.max(np.max(reward_all_normal))
        check_0 = np.min(np.min(reward_all_normal))
        if abs(check_1-1) + abs(check_0-0) < 0.2:
            print('MXairfoil: successfully get weights. \n w='+ str(w) +'\n b=' + str(b))

    def test_time_comsumption(self,N=114514):
        # 
        state0 = self.env.reset()
        total_time_start = time.time()
        for steps in range(N):
            if (steps % 100 == 9):
                print('MXairfoil: episode = '+str(steps))
            action = np.random.uniform(-1.0,1.0,(self.env.real_dim,))
            state, reward, done, _ = self.env.step(action)
        total_time_end = time.time()
        total_time_cost = total_time_end - total_time_start
        print('MXairfoil: total time cost in test_time_comsumption ='+str(total_time_cost))        


if __name__ == '__main__':
    shishi = debug_env()
    # shishi.standardization_performance(N_points=1000) 
    # shishi.get_random_performance()
    # shishi.calculate_weight_performance()

    shishi.standardization_reward(N_points=300,reuse=False) 
    # shishi.calculate_weight_reward()
    for i in range(10):
        shishi.test_time_comsumption(N=1000)
