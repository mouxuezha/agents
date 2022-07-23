# this is to solve the boundary condition.
# main.py is former shishienv**-*.py


import filter_env
from ddpg2 import *
import gc
gc.enable()
from gym import wrappers
import os
from huatu import huatu

# this is for mytest 
import math
import random
import time
import threading 
from multiprocessing import Process

# this is for surrogate.
import sys 
sys.path.append(r'C:/Users/y/Desktop/DDPGshishi/DDPG-master/CDA432-gym/KrigingPython')
from Surrogate_01de import Surrugate

ENV_NAME = 'CDA432_env-v0'
N_threads = 2 
EPISODES = 10000
TEST = 10
dx = 0.1

work_location = 'C:/Users/y/Desktop/DDPGshishi/agents'
log_location = work_location+'/log_for_2dim.txt'
# buffer_location =  work_location+'/buffer/buffer_2dim.pkl'
buffer_location =  work_location+'/agent0_2dim/buffer_2dim.pkl'
agent0_location =  work_location+'/agent0_2dim'
agent1_location =  work_location+'/agent1_2dim'
kaiguan_location = work_location+ '/新建文本文档.txt'
shijian = time.strftime("%Y-%m-%d", time.localtime())
real_obs_space_h = np.array([0.35,-0.22,0.55,8])
real_obs_space_l = np.array([0.25,-0.38,0.35,5])

def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    # agent0 = DDPG(env)
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    agent1 = DDPG2(env,buffer,1)
    # # this is sacrifice for saveing agent0 successfully.  
    agent0 = DDPG2(env,buffer,0)
    # agent1 = DDPG2(env,buffer,1)

    tiaochu=False
    flag0=0
    flag1=0
    saved0 = 0 
    saved1 = 0 

    try:
        buffer.load_buffer(buffer_location)
    except:
        print('MXairfoil: no prepared buffer there ')

    
    try:
        agent0.load_agent(agent0_location)
    except Exception as e:
        print('MXairfoil: no prepared agent0 there ')
        print(e)
        raw_state_new = env.reset_random()*1
        agent0.record_raw_state(raw_state_new)
    # feed the buffer

    while buffer.count() < REPLAY_START_SIZE: 
        bili = (buffer.count()/REPLAY_START_SIZE)*100
        print('MXairfoil: feeding buffer: '+str(bili)+'% ...')
        # feed_buffer_mul(agent0,agent1,ENV_NAME,1)
        feed_buffer_single(agent0,env)

    buffer.save_buffer(buffer_location)
    print('MXairfoil: finish feeding the buffer. Now, start train')
    
    

    # then start the iteration.
    for episode in range(EPISODES):
        #training parallel
        # feed_buffer_mul(agent0,agent1,ENV_NAME,N_threads)
        if (episode % 10 == 9):
            print('MXairfoil: episode = '+str(episode))
        if kaiguan(kaiguan_location)>0:
            print('MXairfoil: forced jump out of a loop')
            break
        
        feed_buffer_single(agent0,env)

        # Testing:
        if (episode % 50 == 49) or (episode<5):
            # agent0.save_agent(agent0_location,episode) # this is for debug, to see untrainned agent.

            flag0,ave_reward0 = average_test(episode,env,agent0,flag0)
            # flag1 = average_test(episode,env,agent1,flag1)
            flag1 = 1 

            # agent0.record_ave_reward(episode,ave_reward0)

            tiaochu = flag0 & flag1 # all are done, then tiaochu
            
            # if flag1 == 0:
            #     agent1.save_agent(agent1_location,episode)
            # elif saved1==0:
            #     agent1.save_agent(agent1_location,episode)
            #     print('MXairfoil: anget1 has been done.')
            #     saved1 =1 


            if flag0 == 0:
                #which means it is still not good enough
                # agent0.save_agent(agent0_location,episode)
                print('MXairfoil: anget0 is not good enough')
                if ave_reward0 > agent0.ave_reward_save0[:,1].max():
                    print('    but is better than last ones')
                    agent0.save_agent(agent0_location,episode)
            elif saved0==0:
                agent0.save_agent(agent0_location,episode)
                print('MXairfoil: anget0 has been done.')
                saved0 =1 

            agent0.record_ave_reward(episode,ave_reward0)
            
        if tiaochu :
            print('MXairfoil: jump out of a loop')
            break
    
    if episode > EPISODES-3 :
        agent0.save_agent(agent0_location,episode)
        print('MXairfoil: EPISODES has been exhausted.')
    env.close()
    huatu_for_main(agent0)
    return agent0 , agent1

def mytest(agent):
    # import random
    print('MXairfoil: start my test...')
    testEnv  = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    print('MXairfoil: test env establised')
    state = testEnv.reset()*1
    dx0 = random.uniform(-1,1)
    next_state,reward,done,_ = testEnv.step(dx0)
    print('MXairfoil: state after a random step ',next_state)
    for i in range(5):
        action = agent.action(next_state)
        print('MXairfoil: action given by trained agent ',action)
        print('MXairfoil: the state here ',next_state)
        next_state,reward,done,_ = testEnv.step(action)
        print('MXairfoil: test env reward is:',reward)
    return 0

def huatu_for_main(agent0):
    tu = huatu(agent0.ave_reward_save0)
    tu.set_location(agent0_location)
    tu.huatu2D('episode','average reward','Reward-Episode relation')
    tu.save_all()

    # draw a picture for raw_state.
    changdu = len(agent0.raw_state_save)
    #raw_state_episode
    shuru = np.append(agent0.raw_state_episode[:,0].reshape(changdu,1),agent0.raw_state_save[:,6].reshape(changdu,1),axis=1)
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

def average_test(episode,env,agent,flag):
    #simplify the code by defining another function.
    # flag = 0 #0 for continue, 1 for good enough.
    if flag == 1 :
        return flag
        #which means this one is  good enough
    kind = agent.kind
    total_reward = 0
    raw_state_new = np.zeros((7,))
    raw_state_performance_new =np.ones((3,))
    # set the constraints here ? or you will test a loneliness 

    for i in range(TEST):
        # state = env.reset_random()*1
        # if agent.raw_state_save[chicun[0] - 1][chicun[1] - 1]<0.7:
        if i<5:
            state = env.reset_random2()*1
            # if it is not good, random search.
        else :
            state = env.reset_random()*1
        # for j in range(env.spec.timestep_limit): 
        for j in range(100):
			#env.render()
            action = agent.action(state) # direct action for test
            state,reward,done,_ = env.step(action)
            total_reward += reward
            performance = env.get_performance()
            # if raw_state_new[raw_state_new.size-1] < state[state.size-1]:
            if raw_state_performance_new[0] > performance[0]: # when new omega is lower, update.
                raw_state_new = state * 1
                raw_state_performance_new = performance* 1
                # get the real 'best' one for uploading.
            
            if done:
                break
    ave_reward = total_reward/TEST

    strbuffer = '\n\nepisode: '+str(episode)+'\nagent id:'+str(kind)+'\nEvaluation Average Reward:'+str(ave_reward) + '\nbuffer size: ' + str(agent.replay_buffer.count())
    jilu(strbuffer)
            
    if (agent.replay_buffer.count() >10000):
        agent.save_buffer(buffer_location)

    if ave_reward>10  or (episode<5) : 
        yuzhi = 0   # yuzhi = 0 for nothing happen 
        # yuzhi = 0.0003 # yizhi != 0 for loosenning the limite and exploring more
        # chicun = agent.raw_state_save.shape
        # if raw_state_new[raw_state_new.size - 1] > (agent.raw_state_save[chicun[0] - 1][chicun[1] - 1]-yuzhi) : # this is reward 
        if raw_state_performance_new[0] < (agent.raw_state_performance[-1][0]-yuzhi) : # this is omega.
            # update only when the newer is better
            env.set_raw_state(raw_state_new)
            agent.record_raw_state(raw_state_new,episode=episode,performance = raw_state_performance_new)
            rizhi = '\nMXairfoil: raw stata updated. \nbefore:'+str(agent.raw_state_save[-2]) + '    performance:'+str(agent.raw_state_performance[-2])+'\nafter'+str(raw_state_new) + '    performance:'+str(raw_state_performance_new)+ '\nartificial tip used number: '+ str(env.N_artificial_tip)
            jilu(rizhi)

    if ave_reward  >52 :
        # this threshold would be changed dynamicly
        print('mytest and save agent,average reward=',ave_reward)
        flag = 1
        print('MXairfoil: it looks there are something relatively good, jump out of a loop')
    return flag,ave_reward

def main_single(i,buffer,ENV_NAME):
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG2(env,buffer,i)
    tiaochu=False
    flag0=0
    saved0 = 0 
    if agent.kind == 0 :
        agent_location = agent0_location
    elif agent.kind == 1:
        agent_location = agent1_location
    
    # then start the iteration.
    
    for episode in range(EPISODES):
        bili = (buffer.count()/REPLAY_START_SIZE)*100
        
        if (buffer.count()<10000)&(episode%100==3):
            print('MXairfoil: feeding buffer: '+str(bili)+'% ...'+'\nthis is thread '+str(i))
        # training parallel
        # feed_buffer_mul(agent0,agent1,ENV_NAME,N_threads)
        state = env.reset()*1
        for step in range(100):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            if agent.kind == 0:
                omega = next_state[agent.action_dim+agent.kind]
                real_reward = 1-5*omega
            elif agent.kind ==1:
                rise = next_state[agent.action_dim+agent.kind]
                real_reward = (rise-1)*10
            agent.perceive(state,action,real_reward,next_state,done)
            # decide what to feed as reward here.
            
            state = next_state
            if done:
                break
        buffer.save_buffer(buffer_location)
        # Testing:
        if (episode % 50 == 1) and (episode>=0) &(buffer.count()>10000):
            flag0 = average_test(episode,env,agent,flag0)

            tiaochu = flag0
            # all are done, then tiaochu
            if (flag0 == 0):
                #which means it is still not good enough
                agent.save_agent(agent_location,episode)
            elif saved0==0:
                agent.save_agent(agent_location,episode)
                print('MXairfoil: anget0 has been done.')
                saved0 =1 
            
        if tiaochu :
            print('MXairfoil: jump out of a loop')
            break
    env.close()

def main_mul(thread_number):
    #just initialize agents in child threads.
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    try:
        # buffer.load_buffer(buffer_location+'buffer'+shijian+'.pkl')
        buffer.load_buffer(buffer_location)
        print('MXairfoil: successfully load  previous buffer')
    except:
        print('MXairfoil: no previous buffer')
    import threading 
    threads = [] 
    nloops=range(thread_number)
    for i in nloops:
        t = threading.Thread(target=main_single,args=(i,buffer,ENV_NAME))
        threads.append(t)
    
    for i in nloops:
        time.sleep(i*3)#avoid same file name.
        threads[i].start()
    # waiting for the end of all threads
    for i in nloops:
        threads[i].join()

def feed_buffer(i,agent,ENV_NAME):
    #this is to feed one group in buffer.
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    state = env.reset()*1
    for step in range(100):
        action = agent.noise_action(state)
        next_state,reward,done,_ = env.step(action)
        agent.perceive(state,action,next_state[4+agent.kind],next_state,done)
        state = next_state
        if done:
            break
    # time.sleep(20)#waiting other process to exit.
    env.close()
    print('MXarifoil: this is thread ',i,' ,finished')

def feed_buffer_mul2(agent0,agent1,ENV_NAME,processe_number):
    # this is trying to feed the buffer in parallel, before before training really start.
    # this is a relatively good version
    # import threading # Importing in every calling is very stupid. 
    # it is said that multiprocessing is more powerful.
    # multiprocessing uses 'spawn' model in windows, it starts a new python.exe and transfer objects into it. the question is, transfering process uses pickle model, but tensorflow objects cannot be 'pickled'
    if sys.platform == 'win32':
        print('MXairfoil: feed_buffer_mul2 is useless on Windows. Get out and modify the code')
        os.system('pause')
    processes = [] 
    nloops=range(processe_number)

    #then build the processes
    for i in nloops:
        t = Process(target=feed_buffer,args=(i,agent0,ENV_NAME))
        
        # if i<round(thread_number/2) :
        #     #when thread_number =3, with out this '-1' , agent1 cannot be traind
        #     t = Process(target=feed_buffer,args=(i,agent0,ENV_NAME))
        # else:
        #     t = Process(target=feed_buffer,args=(i,agent1,ENV_NAME))
         
        # define a processe, and append into processes
        processes.append(t)

    #start the processes 
    for i in nloops:
        time.sleep(i*20)#avoid same file name.
        # keep this even using surrogate model, for it may 
        processes[i].start()
    # waiting for the end of all threads
    for i in nloops:
        processes[i].join()

def feed_buffer_mul(agent0,agent1,ENV_NAME,thread_number):
    # this is trying to feed the buffer in parallel, before before training really start.

    # import threading # Importing in every calling is very stupid. 
    threads = [] 
    nloops=range(thread_number)
    #then build the threads
    for i in nloops:
        t = threading.Thread(target=feed_buffer,args=(i,agent0,ENV_NAME))
        
        # if i<round(thread_number/2) :
        #     #when thread_number =3, with out this '-1' , agent1 cannot be traind
        #     t = threading.Thread(target=feed_buffer,args=(i,agent0,ENV_NAME))
        # else:
        #     t = threading.Thread(target=feed_buffer,args=(i,agent1,ENV_NAME))
         
        # define a thread, and append into threads
        threads.append(t)

    #start the threads 
    for i in nloops:
        time.sleep(i*3)#avoid same file name.
        threads[i].start()
    # waiting for the end of all threads
    for i in nloops:
        threads[i].join()

def feed_buffer_single(agent0,env):
    # this is for surrogate model env. we don't need threading parallel anymore 
    # state = env.reset()*1
    state = env.reset_random()*1
    for step in range(100):
        action = agent0.noise_action(state)
        next_state,reward,done,_ = env.step(action)
        # agent0.perceive(state,action,next_state[4+agent0.kind],next_state,done)
        agent0.perceive(state,action,reward,next_state,done)
        state = next_state
        if done:
            break

def jilu(strBuffer):
    shijian = time.strftime("%Y-%m-%d-%H:%M", time.localtime()) 
    wenjianming = log_location
    rizhi = open(wenjianming,'a')
    rizhi.write(strBuffer)
    rizhi.write('\n'+shijian+'\n')
    rizhi.close()
    print(strBuffer)
    return

def env_test(ENV_NAME):
    # this is to test the new env
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    env_zhen = gym.make(ENV_NAME)
    X_rand = np.random.uniform(0.3,0.7,(2,))
    env_zhen.reset()
    print(env_zhen.state)
    env_zhen.step(X_rand)
    print(env_zhen.state)
    
    env.reset()
    print(env.state)
    env.step(X_rand)
    print(env.state)

    print(env.action_space.shape[0])
    
    print('MXairfoil: finish test env')

def saved_agent_test(**kargs):
    # this is trying to load a saved agent, and get something looks like optimilzed result.
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    if 'agent0' in kargs:
        print('MXairfoil: agent0 inputed')
        agent0 = kargs['agent0']
    else:
        buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        agent1 = DDPG2(env,buffer,1)
        agent0 = DDPG2(env,buffer,0)
        agent0.load_agent(agent0_location)
        agent0.replay_buffer.load_buffer(buffer_location)
    
    if 'constraints' in kargs:
        constraints = kargs['constraints']
    else:
        constraints = np.array([1.0513,35.8])
    try:
        wenjianjia_constraints = str(constraints)
        os.mkdir(agent0_location + '/'+wenjianjia_constraints)
    except:
        print('MXairfoil:fail to make a dir for saved_agent_test')


    test_step = 100
    global_optimization = agent0.raw_state_save[len(agent0.raw_state_save)-1]
    real_go = translate_back(global_optimization)
    rizhi = 'MXairfoil: raw_state reward = ' + str(global_optimization)+'\nin real space:'+ str(real_go) 
    jilu(rizhi)
    kind = agent0.kind
    total_reward = 0
    states_array = np.array([]).reshape(0,7) # this reshape is for appending 
    lujing = np.array([]).reshape(0,7)
    performance = np.array([]).reshape(0,3)

    
    
    for i in range(TEST):
        if i==0:
            state = env.reset()*1
        elif i ==1:
            #get one for real state. # there are some hidden danger here, reset_original is directly called without filter_env.
            state = env.reset_original()*1
        else:
            state = env.reset_random2()*1

        # set constant constraints rather than dynamic 
        # this would overwrite the constraints in env
        state = env.set_constraints(constraints)

        for j in range(test_step):
            state2 = state*1
            lujing = np.append(lujing,translate_surrogate(state2).reshape(1,7),axis=0)
            performance = np.append(performance,env.get_performance().reshape(1,3),axis=0)

            action = agent0.action(state) # direct action for test
            state,reward,done,_ = env.step(action)
            total_reward += reward
            
            if done:
                break
        # save the lujing for further use.
        wenjianming_lujing = agent0_location + '/'+wenjianjia_constraints+'/lujing' + str(i) + '.pkl'
        pickle.dump(lujing,open(wenjianming_lujing,'wb'))
        # save the performance for further use.
        wenjianming_performance = agent0_location + '/'+wenjianjia_constraints + '/performance' + str(i) + '.pkl'
        pickle.dump(performance,open(wenjianming_performance,'wb'))

        lujing = np.array([]).reshape(0,7) # reset this after saved
        performance = np.array([]).reshape(0,3) # reset this after saved

        states_array = np.append(states_array,state.reshape(1,7),axis=0)
        
    ave_reward = total_reward/TEST
    state_optimized = states_array.sum(0)/len(states_array)
    # rizhi = '\n\nMXairfoil: saved_agent_test done. Using this traind agent, optimized state is: ' + str(state_optimized)
    rizhi = '\n\nMXairfoil: saved_agent_test done.'
    jilu(rizhi)
    # shishi = rizhi # test 'communicate by reference' # looks it is base type so 'by value'
    # print('MXairfoil: En Taro XXH ! Long Live mouxuezha! ある学渣万歳！') # cao tai nima shabi le.
    print('MXairfoil: maybe...')
    # os.system('pause')
    return agent0_location + '/'+wenjianjia_constraints

def translate_back(state):
    # state = np.array([ 0.1094045   0.89950771 -0.59411741  0.99998     0.06595014  1.04714399 0.60498643])
    # translate the state back into real parameter space, in a convinient
    # dx = 0.1
    normal_obs_space_h =np.array([1,1,1,1,1,1,1])
    normal_obs_space_l =np.array([-1,-1,-1,-1,-1,-1,-1]) 
    surrogate_obs_space_h = np.array([1+dx,1+dx,1+dx,1+dx])
    surrogate_obs_space_l = np.array([0-dx,0-dx,0-dx,0-dx])
    # real_obs_space_h = np.array([0.4,-0.1,0.8,8])
    # real_obs_space_l = np.array([0.3,-0.5,0.2,3])
    # real_obs_space_h = np.array([0.4,-0.22,0.55,8])
    # real_obs_space_l = np.array([0.3,-0.38,0.35,5])

    # first, transfer from agent/env([-1,1]) into surrogate([0-dx,1+dx])
    bili1 = (surrogate_obs_space_h-surrogate_obs_space_l)/(normal_obs_space_h[0:4] - normal_obs_space_l[0:4])
    zhong_normal = (normal_obs_space_h[0:4] + normal_obs_space_l[0:4])/2
    zhong_surrogate = (surrogate_obs_space_h+surrogate_obs_space_l)/2

    state_surrogate = (state[0:4] - zhong_normal) * bili1 + zhong_surrogate

    # then, transfer from surrogate([0-dx,1+dx]) into real
    # attension, because of virtual grid, this is transfer back form [0,1] to real, despite dx =0.1 
    surrogate_obs_space_h = np.array([1,1,1,1])
    surrogate_obs_space_l = np.array([0,0,0,0])
    zhong_surrogate = (surrogate_obs_space_h+surrogate_obs_space_l)/2
    bili2 = (real_obs_space_h-real_obs_space_l)/(surrogate_obs_space_h - surrogate_obs_space_l)

    zhong_real = (real_obs_space_h+real_obs_space_l) / 2 

    state_real = (state_surrogate - zhong_surrogate) * bili2 + zhong_real
    print('MXairfoil: tranfer back in a cheap way. \n normal state = '+ str(state[0:4])+ '\nsurrogate state = ' + str(state_surrogate) + '\nreal state = ' + str(state_real))
    return  state_real

def translate_surrogate(state):
    normal_obs_space_h =np.array([1,1,1,1,1,1,1])
    normal_obs_space_l =np.array([-1,-1,-1,-1,-1,-1,-1]) 
    surrogate_obs_space_h = np.array([1+dx,1+dx,1+dx,1+dx])
    surrogate_obs_space_l = np.array([0-dx,0-dx,0-dx,0-dx])
    # real_obs_space_h = np.array([0.4,-0.1,0.8,8])
    # real_obs_space_l = np.array([0.3,-0.5,0.2,3])
    # real_obs_space_h = np.array([0.4,-0.22,0.55,8])
    # real_obs_space_l = np.array([0.3,-0.38,0.35,5])

    # first, transfer from agent/env([-1,1]) into surrogate([0,1])
    bili1 = (surrogate_obs_space_h-surrogate_obs_space_l)/(normal_obs_space_h[0:4] - normal_obs_space_l[0:4])
    zhong_normal = (normal_obs_space_h[0:4] + normal_obs_space_l[0:4])/2
    zhong_surrogate = (surrogate_obs_space_h+surrogate_obs_space_l)/2

    state_surrogate = (state[0:4] - zhong_normal) * bili1 + zhong_surrogate

    state[0:4] = state_surrogate
    return state

def real_to_norm_state(state):
    # from real to [0-dx,1+dx], then from [0-dx,1+dx] to [-1,1]
    # dx = 0.0
    normal_obs_space_h =np.array([1,1,1,1,1,1,1])
    normal_obs_space_l =np.array([-1,-1,-1,-1,-1,-1,-1]) 
    # real_obs_space_h = np.array([0.4,-0.1,0.8,8])
    # real_obs_space_l = np.array([0.3,-0.5,0.2,3])
    # real_obs_space_h = np.array([0.4,-0.22,0.55,8])
    # real_obs_space_l = np.array([0.3,-0.38,0.35,5])
    surrogate_obs_space_h = np.array([1,1,1,1])
    surrogate_obs_space_l = np.array([0,0,0,0])

    real_state_bili = (real_obs_space_h-real_obs_space_l)/(surrogate_obs_space_h - surrogate_obs_space_l)
    real_state_c = (real_obs_space_h + real_obs_space_l)/2

    surrogate_state_bili = ( surrogate_obs_space_h - surrogate_obs_space_l ) /(normal_obs_space_h[0:4] - normal_obs_space_l[0:4]) 
    surrogate_state_c = ( surrogate_obs_space_h + surrogate_obs_space_l ) /2
    
    surrogate_state = (state - real_state_c) / real_state_bili + surrogate_state_c 

    # this is where the differents are
    surrogate_obs_space_h = np.array([1+dx,1+dx,1+dx,1+dx])
    surrogate_obs_space_l = np.array([0-dx,0-dx,0-dx,0-dx])

    surrogate_state_bili = ( surrogate_obs_space_h - surrogate_obs_space_l ) /(normal_obs_space_h[0:4] - normal_obs_space_l[0:4]) 
    surrogate_state_c = ( surrogate_obs_space_h + surrogate_obs_space_l ) /2

    normal_state_c = (normal_obs_space_h[0:4] + normal_obs_space_l[0:4])/2
    norm_state = (surrogate_state - surrogate_state_c) / surrogate_state_bili + normal_state_c
    return surrogate_state,norm_state

def surrogate_to_norm_state(surrogate_state):
    # from [0-dx,1+dx] to [-1,1]
    # dx = 0.0
    normal_obs_space_h =np.array([1,1,1,1,1,1,1])
    normal_obs_space_l =np.array([-1,-1,-1,-1,-1,-1,-1]) 
    surrogate_obs_space_h = np.array([1,1,1,1])
    surrogate_obs_space_l = np.array([0,0,0,0])

    # this is where the differents are
    surrogate_obs_space_h = np.array([1+dx,1+dx,1+dx,1+dx])
    surrogate_obs_space_l = np.array([0-dx,0-dx,0-dx,0-dx])

    surrogate_state_bili = ( surrogate_obs_space_h - surrogate_obs_space_l ) /(normal_obs_space_h[0:4] - normal_obs_space_l[0:4]) 
    surrogate_state_c = ( surrogate_obs_space_h + surrogate_obs_space_l ) /2

    normal_state_c = (normal_obs_space_h[0:4] + normal_obs_space_l[0:4])/2
    norm_state = (surrogate_state - surrogate_state_c) / surrogate_state_bili + normal_state_c
    return norm_state

def kaiguan(location):
    # this is a simple on-off. if the location is no exists, then nothing would happen, if I buid it, then main() should end.
    if os.path.exists(location) :
        zhi = 1 # for end the program
        print('MXairfoil: end the program by hand')
    else :
        zhi = 0 # for nothing happen
    return zhi

def huatu_post(loaction):
    tu = huatu(0)
    # huatu for agent 2d.
    tu.set_location(loaction)
    # tu.visual_2D(1,0) 
    # tu.visual_2D(2,0)
    for i in range(10):
        tu.visual_2D(4,i) 


if __name__ == '__main__':
    # agent0 , agent1 = main()
    # check two 'dx's, raw_state, and flag before start, along with the kaiguan state , log, and clear or not.
    flag = 5
    if flag ==0:
        agent0 , agent1 = main()
        mulu =saved_agent_test(agent0=agent0)
        huatu_post(mulu)
    elif flag ==1:
        env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
        buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        # agent0 = DDPG2(env,buffer,0)
        agent1 = DDPG2(env,buffer,2)
        agent0 = DDPG2(env,buffer,0)
        agent1.save_agent(agent1_location,0)
        agent0.save_agent(agent0_location,0)
        del agent1
        agent1 = DDPG2(env,buffer,1)
        agent1.save_agent(agent1_location,0)
    elif flag == 3:
        #this is for testing the env 
        env_test(ENV_NAME)
    elif flag == 4:
        #this is for test the buffer
        buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        buffer.save_buffer(buffer_location+'/shishibuffer.pkl')
        buffer.load_buffer(buffer_location+'/shishibuffer.pkl')
    elif flag == 5:
        # this is to test a saved agent.
        print('MXairfoil: this is to test a saved agent.')
        constraints = np.array([[1.0510,34.2],[1.0517,37.1],[1.0513,35.8]])
        mulu = saved_agent_test(constraints=constraints[1])
        huatu_post(mulu)
        # for i in range(len(constraints)):
        #     mulu = saved_agent_test(constraints=constraints[i])
        #     huatu_post(mulu)

    elif flag == 7:
        # this is to test the kaiguan()
        zhi = 0
        while zhi <0.5:
            time.sleep(1)
            print('MXairfoil: test stop')
            zhi = kaiguan(kaiguan_location)
    elif flag == 8:
        # this is to test huatu
        huatu_for_main()
    print('MXairfoil: all finish, En Taro XXH!')
    
