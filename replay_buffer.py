from collections import deque
import os
import random
import pickle
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size,**kargs):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

        if 'state_dim' in kargs:
            self.state_dim = kargs['state_dim']
        else:
            self.state_dim = 7 
        if 'action_dim' in kargs:
            self.action_dim = kargs['action_dim']
        else:
            self.action_dim = 2 


    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        # self.jilu_buffer(experience)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

    def save_buffer(self,location):
        # pickle.dump(self.buffer,open('C:/Users/y/Desktop'),'w')
        pickle.dump(self.buffer,open(location,'wb'))

    def load_buffer(self,location):
        # self.num_experiences = 0
        self.buffer = deque()

        try:
            self.buffer = pickle.load(open(location,'rb'))
        except:
            print('MXairfoil: no prepared buffer')
        self.num_experiences = len(self.buffer)

    
    def load_buffer_txt(self,wenjianming):
        print('MXairfoil: this fuction has been freeze, go and check the code ')
        os.system("pause")
        # wenjianming = 'C:/Users/y/Desktop/EnglishMulu/testCDA1/main/log2021-04-13data.txt'
        file = open(wenjianming,'r')
        
        lines = file.readlines()
        length = len(lines)
        data = np.zeros((length, 6))
        state = np.zeros((length, 4))
        index = 0 
        for line in lines:
            line = line.strip('[')
            line = line.strip(']\n')
            shuzi = line.split(',')
            print(shuzi)
            data[index,:] = shuzi[:]
            index += 1 
            state[index,:] = data[index,0:4]
        file.close()
        return data

    def jilu_buffer(self,experience):
        print('MXairfoil: this fuction has been freeze, go and check the code ')
        os.system("pause")
        wenjianming = 'C:/Users/y/Desktop/DDPGshishi/buffer.txt'
        line = str(experience)
        line = line.strip('(')
        line = line.strip(')')
        line = line.split('array[')
        shishi = line[0]
        shishi = shishi.replace('array([',' ')
        shishi = shishi.replace('])',' ')
        # data = np.zeros((1, state.size+new_state.size+action.size+2))
        data = np.zeros((1, (4+4+2+2)))
        shuzi = shishi.split(',')
        data[0,:] = shuzi[:]
        file=open(wenjianming,'a')
        file.write(str(data[0]))
        file.write('\n')
        file.close()

if __name__ == '__main__':
    #learn and test buffer.
    print('MXairfoil: learn and test buffer.')
    shishi  = ReplayBuffer(1000)
    data = shishi.load_buffer_txt('C:/Users/y/Desktop/EnglishMulu/testCDA1/main/log2021-04-13data.txt')


