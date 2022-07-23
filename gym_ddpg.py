import filter_env
from ddpg import *
import gc
gc.enable()
from gym import wrappers

# ENV_NAME = 'InvertedPendulum-v1'
# there are KeyError. try some modify.
# ENV_NAME = 'CartPole-v0' #RuntimeError: Environment with continous action space (i.e. Box) required.
ENV_NAME = 'InvertedPendulum-v2'
# http://gym.openai.com/envs/#mujoco

# ENV_NAME = 'shishi_env-v0'

EPISODES = 100000
TEST = 10

def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)
    # env.monitor.start('experiments/' + ENV_NAME,force=True) # this is something related to gym, not tensorflow. https://www.programcreek.com/python/example/100947/gym.wrappers.Monitor
    # env = wrappers.Monitor(env,'experiments/' + ENV_NAME,force=True)
    # I'm asked to use tab in python, or it will report "Inconsistent use of tabs and spaces in indentation"
    # there are no xrange in python 3.8.3, I will use range instead.
    for episode in range(EPISODES):
        state = env.reset()
        #print "episode:",episode
        # Train
        # for step in range(env.spec.timestep_limit):
        for step in range(1000):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % 100 == 0 and episode > 100:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                # for j in range(env.spec.timestep_limit): 
                for j in range(1000):
					#env.render()
                    action = agent.action(state) # direct action for test
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward/TEST
            print( 'episode: ',episode,'Evaluation Average Reward:',ave_reward)
        # env.monitor.close()
        env.close()

if __name__ == '__main__':
    main()
