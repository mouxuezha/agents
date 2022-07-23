# En Taro XXH!

from turtle import Turtle
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

from main_auto import auto_jisuan
# if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
#     # which means in my diannao.
#     sys.path.append(r'E:\EnglishMulu\KrigingPython')
# else:
#     sys.path.append(r'D:\XXHdatas\KrigingPython')
# from Surrogate_01de import Surrugate
# from Surrogate_01de import record_progress # it must be here, or class record_progress can not be found.
# import Rotor67_gym


if __name__ == '__main__':
    total_time_start = time.time()
    flag =43
    # 0 for huatu, 1 for run.
    # 2 for new democase of 18 dim.
    # 3 for Rotor67 case.
    # 4 for the renaissance of CDA case.
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
    elif flag == 3 : 
        if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
            # which means in my diannao.
            sys.path.append(r'E:\EnglishMulu\KrigingPython')
        else:
            sys.path.append(r'D:\XXHdatas\KrigingPython')
        from Surrogate_01de import Surrugate
        from Surrogate_01de import record_progress # it must be here, or class record_progress can not be found.
        import Rotor67_gym        
        # trainning Rotor67 case.
        shishi = auto_jisuan(dx=0.1,feed_back_enable=True,ENV_NAME = 'Rotor67_env-v0',ave_reward_threshold=70,buffer_reuse=False,EPISODES=8000,agent_reuse = False,zuobi=False,episode_min=3998)
        # shishi.main()
        shishi.auto_run2(jvli_threshold=0.04,guanghua_threshold=0.995,auto_steps=30,model='simple',episode_batch_main=50) 
    elif flag == 31 : 
        # debuging Rotor67 case.
        shishi = auto_jisuan(dx=0.1,feed_back_enable=True,ENV_NAME = 'Rotor67_env-v0',ave_reward_threshold=70,buffer_reuse=True,EPISODES=8000,agent_reuse = True,zuobi=False,episode_min=3998)
        shishi.main()
        # shishi.auto_run2(jvli_threshold=0.04,guanghua_threshold=0.995,auto_steps=30,model='simple',episode_batch_main=50) 
    elif flag == 32 :
        # recycle a trained agent.
        shishi = auto_jisuan(dx=0.1,feed_back_enable=True,ENV_NAME = 'Rotor67_env-v0',ave_reward_threshold=68,buffer_reuse=False,EPISODES=6000,agent_reuse = False,zuobi=False,episode_min=4500)
        shishi.recycle_exist_agent(jvli_threshold=0.04,guanghua_threshold=0.995)
    elif flag == 41 :
        # debuging CDA 2D case. updated from DDPG-master82
        sys.path.append(r'C:/Users/y/Desktop/DDPGshishi/DDPG-master/CDA42-gym/KrigingPython')
        from Surrogate_01de import Surrugate
        import CDA42_gym
        shishi = auto_jisuan(dx=0.1,feed_back_enable=True,ENV_NAME ='CDA42_env-v0' ,ave_reward_threshold=70,buffer_reuse=True,EPISODES=1500,agent_reuse = False,zuobi=False,episode_min=100)
        # shishi.main()
        shishi.auto_run2(jvli_threshold=0.04,guanghua_threshold=0.995,auto_steps=30,model='simple',episode_batch_main=50) 
    elif flag == 42:
        # debuging CDA 2D case. updated from DDPG-master83
        sys.path.append(r'C:/Users/y/Desktop/DDPGshishi/DDPG-master/CDA43-gym/KrigingPython')
        from Surrogate_01de import Surrugate
        import CDA43_gym
        shishi = auto_jisuan(dx=0.1,feed_back_enable=True,ENV_NAME ='CDA43_env-v0' ,ave_reward_threshold=56,buffer_reuse=True,EPISODES=800,agent_reuse = False,zuobi=False,episode_min=500)
        # shishi.main()
        shishi.auto_run2(jvli_threshold=0.04,guanghua_threshold=0.995,auto_steps=30,model='simple',episode_batch_main=25) 
    elif flag == 43:
        # debuging CDA 4D case. updated from DDPG-master84
        sys.path.append(r'C:/Users/y/Desktop/DDPGshishi/DDPG-master/CDA44-gym/KrigingPython')
        from Surrogate_01de import Surrugate
        import CDA44_gym
        shishi = auto_jisuan(dx=0.1,feed_back_enable=True,ENV_NAME ='CDA44_env-v0' ,ave_reward_threshold=80,buffer_reuse=True,EPISODES=5000,agent_reuse = False,zuobi=False,episode_min=600)
        # shishi.main()
        shishi.auto_run2(jvli_threshold=0.04,guanghua_threshold=0.995,auto_steps=30,model='simple',episode_batch_main=25) 


    elif flag == -999 :
        shishi = auto_jisuan(dx=0.1,feed_back_enable=True,ENV_NAME = 'Demo180_env-v0',ave_reward_threshold=50,buffer_reuse=True,EPISODES=1000,agent_reuse = False,zuobi=False)
        shishi.clear_agents(clear_storage=True)        
    total_time_end = time.time()
    total_time_cost = total_time_end - total_time_start
    print('MXairfoil: total time cost ='+str(total_time_cost))