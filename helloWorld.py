# this is for showing the inderactive window.

import numpy as np 
import time 
import os 

from main_auto import auto_jisuan
import psutil
import pickle

# import sys
# sys.path.append(r'C:/Users/y/Desktop/DDPGshishi/DDPG-master/CDA3-gym/KrigingPython')
# from Surrogate_01de import Surrugate
class debug_env(auto_jisuan):
    def __init__(self) -> auto_jisuan:
        super().__init__(ENV_NAME = 'Rotor67_env-v0',dx=0.1)
        
        pass

    def standardization_performance(self,N_points = 114514):
        # calculate several steps to get perfromance_w and performance_b 
        performance_all = np.zeros((self.env.performance_dim,N_points))
        for i in range(N_points):
            # caonima
            performance_all[i] = self.get_random_performance(self.env)
        weizhi = self.work_location + '/performance_all.pkl'
        # self.X = pickle.load(open(location_X,'rb'))
        pickle.dump(performance_all,open(weizhi,'wb'))

    def get_random_performance(self,env):
        env.reset()
        performance = env.get_performance()
        return performance

    def calculate_weight(self):
        # map the performance into [0,1]
        weizhi = self.work_location + '/performance_all.pkl'
        if self.performance_all:
            self.performance_all = pickle.load(open(weizhi,'rb'))
        performance_min = np.min(self.performance_all,axis=0)
        performance_max = np.max(self.performance_all,axis=0)
        performance_cha = performance_max - performance_min
        
        w = 1.0 / performance_cha
        b = 0 - w*performance_min
        performance_all_normal = w*self.performance_all + b
        check_1 = max(max(performance_all_normal))
        check_0 = min(min(performance_all_normal))
        if abs(check_1-1) + abs(check_0-0) < 0.00000001:
            print('MXairfoil: successfully get weights. w = '+ str())


if __name__ == '__main__':

    print('MXairfoil: interactive window establised.')

    theta = np.random.uniform(0,2*np.pi,(4-1,))
    r = 1
    # dstate = np.array([0,0,0,0], dtype=float)
    dstate = np.array([0.,0.,0.,0.])
    dstate[0] = r 
    for i in range(4-1): # basic operation, sit down and no 666 please.
        zhongjie = dstate[i] * 1
        dstate[i] = zhongjie*np.cos(theta[i])
        dstate[i+1] = zhongjie*np.sin(theta[i])
    check = dstate**2
    check = check.sum()
    print(check)
