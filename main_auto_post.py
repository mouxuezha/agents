# post process for main_auto.


import filter_env
# from ddpg2 import *
from ddpg_shishi import *
from gym import wrappers
import os
import shutil
from huatu import huatu
from huatu import Pjudge
import time
from multiprocessing import Process
from transfer import transfer
import random
import sys 

from parameters import parameters_Rotor67_state

from main_auto import auto_jisuan
if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
    # which means in my diannao.
    sys.path.append(r'E:\EnglishMulu\KrigingPython')
else:
    sys.path.append(r'D:\XXHdatas\KrigingPython')
from Surrogate_01de import Surrugate
from Surrogate_01de import record_progress # it must be here, or class record_progress can not be found.
import Rotor67_gym

class main_auto_post(auto_jisuan):
    def __init__(self,ENV_NAME = 'Rotor67_env-v0',dx=0.1,weizhi = None) :
        # import Demo180_gym # just shishi if it is useful here.
        super().__init__(ENV_NAME = ENV_NAME,dx=dx)
        self.N_lujing = 10 
        self.folder = weizhi
        self.state_original_real = np.array( [0,0,0,0,0,0,0.447895,0.122988,0.064253,0.050306, -0.639794,-0.052001,-0.050454,-0.148836,0.223533,0.656313,0.965142,1.098645])
        self.method = 'undefined'
        pass

    def get_original_state(self):
        state_original_real = self.env.state_original_real
        state_original_normal = self.env.transfer_obs.real_to_normal(state_original_real)
        return state_original_normal

    def get_state_performance(self,state_input):
        # get everything for one state.state_input should in [-1,1]
        self.env.reset()
        state_output, reward, done, asd = self.env.set_state(state_input) 
        performance = self.env.get_performance() *1.0
        return performance
    
    def get_original_performance(self):
        performance = np.array([0.93027085, 0.98301396, 0.87772323, 1.64442443, 0.96573699,0.88285   , 1.644     ])
        performance = performance.reshape(1,7)
        return performance

    def load_lujing(self,location=None):
        if location == None:
            location = self.agent0_location
        
        lujing = np.array([]).reshape(0,self.real_dim)
        performance = np.array([]).reshape(0,self.env.performance_dim)

        lujing_list = [] 
        performance_list = []
        
        for i in range(self.N_lujing):
            wenjianming_lujing = location +'/lujing' + str(i) + '.pkl' 
            wenjianming_performance = location + '/performance' + str(i) + '.pkl'

            lujing = pickle.load(open(wenjianming_lujing,'rb'))
            performance = pickle.load(open(wenjianming_performance,'rb'))

            lujing_list.append(lujing)
            performance_list.append(performance)
        
        return lujing_list , performance_list

    def get_agent_result(self,panduan,location=None,model='normal'):
        # load lujing, calculate end point, average, get state, and compare with
        lujing_list,performance_list = self.load_lujing(location)

        flag_panju,strbuffer_Pjudge = panduan.Pjudge_test(lujing_list,model = 'str_on')

        # 
        s_dim = self.real_dim
        p_dim = self.env.performance_dim
        s_end = np.zeros([self.N_lujing,s_dim])
        p_end = np.zeros([self.N_lujing,p_dim])
        for i in range(self.N_lujing):
            s_end[i] = lujing_list[i][len(lujing_list[i])-1,0:s_dim]
            p_end[i] = performance_list[i][len(lujing_list[i])-1,0:p_dim]

        s_end_ave = np.mean(s_end,axis=0) 
        s_end_ave_normal = self.env.transfer_obs.surrogate_to_normal(s_end_ave)

        p_end_ave = np.mean(p_end,axis=0) 
        p_end_new = self.get_state_performance(s_end_ave_normal)

        p_end_check =np.linalg.norm( p_end_ave - p_end_new)
        if (p_end_check > 0.01*p_dim) and (self.env.kriging_flag):
            raise Exception('MXairfoil: it seems something wrong in performance')

        # then get original
        state_original_normal = self.get_original_state()
        performance_original_normal = self.get_state_performance(state_original_normal)
        # performance_original_normal = self.get_original_performance()

        # then objective function:
        index_objective = 3 
        obj_before = performance_original_normal[index_objective]
        obj_after = p_end_new[index_objective]
        bili = (obj_after - obj_before) / obj_before
        strbuffer = '\n============================\nMXairfoil: objective funtion index is '+ str(index_objective)+ ' \nbefore: ' + str(obj_before) + '\nafter:'+ str(obj_after) + '\nincrease(%):' + str(bili*100) + '\noriginal performance:' + str(performance_original_normal) + '\nnew performance' + str(p_end_new)  + '\nend state:'+str(s_end_ave_normal) + '\noriginal state:'+str(state_original_normal)  + '\nlocation' + location +'\nPjudge:' + strbuffer_Pjudge
        # print(strbuffer)
        if model == 'normal':
            self.log_location = self.folder+'/说明.txt'
            self.jilu(strbuffer)
        elif model == 'get_state':
            print(strbuffer)
            return s_end_ave_normal

    def get_agent_result_auto(self,folder = None,method = 'DDPG'):
        # auto test all results in one location.
        index = 0 
        jvli_threshold=0.04
        guanghua_threshold=0.995
        panduan = Pjudge(dim=self.real_dim,jvli_threshold=jvli_threshold,guanghua_threshold=guanghua_threshold)
        if folder==None:
            folder = self.storage_location
        
        for index in range(114514) : 
            location_target = folder + self.get_agent_name(method)+str(index)  
            if os.path.exists(location_target):
                self.get_agent_result(panduan,location=location_target)
            else:
                print('MXairfoil: no such file:\n'+location_target)
                break
        
        print('MXairfoil: finish getting result ')

    def huatu_slice(self,location=None,x_index=0,y_index=1,data_id=[1,2,3],index=0,name=None):
        # select two dim and draw picture.+
        lujing_list,performance_list = self.load_lujing(location)
        tu_name = 'visual_2D_simplified'+'_' + str(index) + name
        label = [] 
        for i in range(len(data_id)):
            label.append('number '+str(data_id[i])+' process')
        tu=huatu(0)
        tu.visual_2D_simplified(lujing_list,data_id = data_id,x_index=x_index,y_index=y_index,meiju = parameters_Rotor67_state,tu_name = tu_name , location = location,label=label,adjust=[0.2,0.95,0.2,0.95])

    def get_agent_name(self,method):
        if self.method != method:
            print('MXairfoil: method changed, attenstion.')
            self.method = method
        if method == 'DDPG':
            location_target ='/agent0indedx'
        elif method == 'PPO':
            location_target = '/agent0PPOindedx'
        else:
            raise Exception('MXairfoil: invalid method')  
        return location_target
    def huatu_slice_auto(self,folder=None):
        index = 0 
        if folder==None:
            folder = self.storage_location
        for index in range(114514) : 
            location_target = folder + self.get_agent_name(self.method)+str(index)
            if os.path.exists(location_target):
                self.huatu_slice(location = location_target,x_index=0,data_id=[0,2,4,6,8],y_index=4,index=index,name=self.method)
            else:
                print('MXairfoil: no such file:\n'+location_target)
                break
        
        print('MXairfoil: finish huatu_slice ')

    def huatu_slice_detail(self,folder=None,index=0,data_id=[0,2,4,6]):
        # hua some slice for a relatively good agent.
        location_target = folder + '/agent0indedx'+str(index)  

        x_index = 0 #span_dm
        y_index = 4 #span_dtheta
        self.huatu_slice(location = location_target,x_index=x_index,data_id=data_id,y_index=y_index,index=index,name='span')

        x_index = 7 
        y_index = 11 
        self.huatu_slice(location = location_target,x_index=x_index,data_id=data_id,y_index=y_index,index=index,name='chi')

        x_index = 7 
        y_index = 15
        self.huatu_slice(location = location_target,x_index=x_index,data_id=data_id,y_index=y_index,index=index,name='zeta')

    def huatu_for_history(self,folder=None,**kargs):
        # huatu for history. Many lines in gray and one in black.
        # this function should have the capability of directly running without trainning process.
        if 'converge_history' in kargs:
            converge_history = kargs['converge_history']
        else:
            converge_history = []
        if 'converge_history_opt' in kargs:
            converge_history_opt = kargs['converge_history_opt']
        else:
            converge_history_opt = []

        # converge_history = []
        # converge_history_opt = []
        if folder==None:
            folder = self.storage_location

        if len(converge_history) == 0 :
            # no input, so load from storage
        
            wenjianming = folder + '/Converge History/converge_history.pkl' 
            wenjianming_opt = folder + '/Converge History/converge_history_opt.pkl' 
            index = 0 
            
            for index in range(114514) : 
                location_target = folder + '/agent0indedx'+str(index) 
                wenjianming_x = location_target + '/Reward-Episode relation/x.pkl' 
                wenjianming_y = location_target + '/Reward-Episode relation/y.pkl' 
                if os.path.exists(wenjianming) and os.path.exists(wenjianming_opt):
                    converge_history = pickle.load(open(wenjianming,'rb'))
                    converge_history_opt = pickle.load(open(converge_history_opt,'rb'))
                    print('MXairfoil: converge_history loaded:\n'+wenjianming + '\nconverge_history loaded: \n' + wenjianming_opt)
                    break
                elif os.path.exists(wenjianming_x):
                    converge_history_x = pickle.load(open(wenjianming_x,'rb'))
                    converge_history_y = pickle.load(open(wenjianming_y,'rb'))
                    converge_history_x = converge_history_x.astype(float)
                    converge_history_y = converge_history_y.astype(float)
                    converge_history_single = [converge_history_x,converge_history_y]
                    converge_history.append([converge_history_single])
                    print('MXairfoil: converge_history loaded sperately:\n'+location_target)
                else:
                    print('MXairfoil: no such file:\n'+location_target)
                    break

            # then, get mean line for converge_history.
            mean_line_num = 114514
            mean_line_num_max = 0 
            for converge_history_single in converge_history:
                mean_line_num = min(mean_line_num,len(converge_history_single[0][0]))
                # mean_line_num_max = max(mean_line_num_max,len(converge_history_single[0][0]))
                if len(converge_history_single[0][0]) > mean_line_num_max:
                    mean_line_num_max = len(converge_history_single[0][0])
                    mean_line_x_max = converge_history_single[0][0]
            mean_line_y = np.zeros(mean_line_num_max)
            mean_line_y_num = np.zeros(mean_line_num_max) 
            # for converge_history_single in converge_history:
            #     mean_line_y[0:mean_line_num] = mean_line_y[0:mean_line_num] + converge_history_single[0][1][0:mean_line_num]
            # mean_line_y[0:mean_line_num] = mean_line_y[0:mean_line_num] / mean_line_num
            # mean_line_x = converge_history[0][0][0][0:mean_line_num]
            # mean_line = [mean_line_x,mean_line_y]
            for converge_history_single in converge_history:
                changdu = len(converge_history_single[0][1])
                mean_line_y[0:changdu] = mean_line_y[0:changdu] + converge_history_single[0][1][0:changdu]
                mean_line_y_num[0:changdu] = mean_line_y_num[0:changdu] + 1
            
            mean_line_y = mean_line_y / mean_line_y_num
            mean_line = [mean_line_x_max,mean_line_y]
            print('MXairfoil: finish load  converge_history')


        tu = huatu(0) # it is assumed that the last one is the best.
        tu.set_location(folder) # this is for all converge history, rather than one round trainning.
        tu.load_data_add(converge_history[-1])
        # tu.load_data_add(converge_history_opt[-1])
        tu.huatu2D_mul2('Episode','Total reward','Converge History','100 step total reward',modle='all',single_ax=True)
        
        # tu.huatu2D_add_grey_line(converge_history_opt[-1],zhonglei='opt_chosen')

        # then draw the normal lines.
        for line_single in converge_history:
            tu.huatu2D_add_grey_line(line_single[0],zhonglei='ave_normal')
        # for line_single in converge_history:
        #     tu.huatu2D_add_grey_line(line_single,zhonglei='ave_normal')
       
        tu.huatu2D_add_grey_line(mean_line,loc='upper left',zhonglei='mean_line',label='mean line')
        tu.save_all()

    def get_state_end(self,folder=None,index=0,return_model = 'surrogate'):
        # very simple, just get a state end of this one
        jvli_threshold=0.04
        guanghua_threshold=0.995  
        location_target = folder + '/agent0indedx'+str(index)  
        panduan = Pjudge(dim=self.real_dim,jvli_threshold=jvli_threshold,guanghua_threshold=guanghua_threshold)      
        zhi = self.get_agent_result(panduan,location=location_target,model='get_state')
        
        # state_end_real = self.transfer.surrogate_to_real(zhi)
        state_end_real = self.transfer.normal_to_real(zhi)
        state_end_normal = self.transfer.real_to_normal(state_end_real)
        state_end_surrogate = self.transfer.normal_to_surrogate(state_end_normal)
        
        # state_original_surrogate = self.transfer.real_to_surrogate(self.state_original_real)
        # self.state_original_real2 = self.transfer.surrogate_to_real(state_original_surrogate)
        # state_original_normal = self.transfer.real_to_normal(self.state_original_real)
        
        real_obs_range = self.real_obs_space_h-self.real_obs_space_l


        dstate = abs(abs(state_end_real) - abs(self.state_original_real))
        dstate_rate = dstate / real_obs_range
        print('\n================================\nMXairfoil: state_end of this agent is:  \n' + str(state_end_real)+ '\n dstate rate is: \n' + str(dstate_rate))
        if return_model == 'surrogate':
            return state_end_surrogate
        elif return_model == 'normal':
            return state_end_normal
        elif return_model == 'real':
            return state_end_real
    def storage_agent_test_auto(self,folder=None,index=None):
        # calculate lujing again.
        if not(self.init_agents_flag):
            self.init_agents()
        if index ==None:
            for index in range(114514) : 
                location_target = folder + '/agent0indedx'+str(index) 
                if os.path.exists(location_target):
                    self.storage_agent_test(location_target,index)
                else:
                    print('MXairfoil: no such file:\n'+location_target)
                    break 
        else:
            location_target = folder + '/agent0indedx'+str(index)  
            self.storage_agent_test(location_target,index)
    def storage_agent_test(self,location_target,index):
        self.agent0.reset_agent(location =self.agent0_location) 
        shutil.rmtree(self.agent0_location) 
        shutil.copytree(location_target,self.agent0_location)
        self.saved_agent_test()
        self.save_agents_result(self.agent0_location,index=index) 

if __name__ == '__main__':
    
    # weizhi = r'E:\EnglishMulu\agents\Rotor67的第一波'
    # weizhi = r'E:\EnglishMulu\agents\Rotor67的第二波'
    # weizhi = r'E:\EnglishMulu\agents\Rotor67的第三波'
    weizhi = r'E:\EnglishMulu\agents_PPO'
    # weizhi = r'E:\EnglishMulu\agents\二维无约束'

    flag =0
    if flag ==0:
        # standared post process, for a folder.
        shishi = main_auto_post(weizhi=weizhi)
        # shishi.get_agent_result_auto(weizhi,method='PPO')
        shishi.get_agent_result_auto(weizhi,method='DDPG')
        shishi.huatu_slice_auto(weizhi)
        shishi.huatu_for_history(folder=weizhi)
    elif flag == 1:
        # detailed huatu for one of the agents 
        shishi = main_auto_post(weizhi=weizhi)
        shishi.huatu_slice_detail(folder=weizhi,index=15,data_id=[0,2,4,6,8])
        shishi.get_state_end(folder=weizhi,index=15)
    elif flag == 2:
        # recycle from storage.
        shishi = main_auto_post(weizhi=weizhi)
        shishi.storage_agent_test_auto(folder=weizhi,index=15)
    elif flag == 999:
        # for debug
        shishi = main_auto_post(weizhi=weizhi)
        shishi.get_state_end(folder=weizhi,index=15)