#for defining dqn class

import os
import sys
import collections
import random
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
action은 값을 여러개 설정할 필요 없이, +1 or -1 or 0으로 해서 대입하면 된다
-> 1개의 node에서 나오는 continuous output을 적당히 계산해서 action으로 변형
or sigmoid 사용하기

따라서 Qnet의 output node는 총 4개, 각각이 h1, v1, h2, v2의 motor value 더하거나 빼는 것
전송할 때도 4bit 형태로 통신 가능

++ 관절 2개로는 학습이나 유의미한 결과가 나타나기 어려워보임
-> 관절 한개만 더 추가해서 총 3관절 만들기 -> 6개 output
'''

# Hyperparameters
learning_rate = 0.0001
gamma = 0.98
buffer_limit = 50000
batch_size = 32
train_start = 5000

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

#Replay buffer
class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit) # queue 선언

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n): #randomly sample n
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition #state, action, reward, next state, done
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_mask_lst) #sampling한 torch tensor들을 return

    def size(self):
        return len(self.buffer)

#Qnet
class Qnet(nn.Module): 
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 256) # sensor 값에 따른 input 결정
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 6) #servo 값 결정 -> 실수로 출력 가능?

    def forward(self, x): #get the Q network result (state to action, policy)
        x = F.sigmoid(self.fc1(x)) #sigmoid can be used
        x = F.sigmoid(self.fc2(x)) # original relu
        x = F.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon, memory_size): # epsilon greedy 반영하여 action하는 것... 
        if memory_size < train_start: #random action
            return random.randint(-1, 1, size=4)
            '''
            여기서부터 더 수정
            '''
        else:
            out = self.forward(obs) # action determined by self forwarding function & observation
            coin = random.random()
            if coin < epsilon: #epsilon greedy method!
                return random.randint(0, 4)
            else:
                return out.argmax().item() #argmax -> index 반환

    def action(self, obs): #action case
        out = self.forward(obs)
        return out.argmax().item()

    def action_to_stearing(self, a):
        steer = (a - 2) * (np.pi /30) #modified to 3 -> 30degree
        if a == 2:
            speed = 9.8 #5.0 -> 8.0 -> 9.8
        elif a == 1 or a == 3:
            speed = 8.5 # 6.5 -> 8.5
        else:
            speed = 8.0 #6.0 -> 8.0
        return [steer, speed]

    def preprocess_lidar(self, ranges):
        eighth = int(len(ranges) / 8)

        return np.array(ranges[eighth:-eighth: 2])

#train -> 수정 필요
'''
state: robot의 현재 각도 정보
action: +1 or -1로 결정
reward: sensor의 값 가지고 python code에서 계산 및 결정
done: timeout? 말고 어떤 done 유형이 있을까 -> 따로 없어보임
'''
def train(q, q_target, memory, optimizer):

    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        q_out = q(s) # q network에 state = observation 대입, action 취함
        q_a = q_out.gather(1, a) #action 추출
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask # action value update
        loss = F.smooth_l1_loss(q_a, target)
        '''
        loss function 정의할 필요가 있음!!
        '''

        optimizer.zero_grad()
        loss.backward() # backprop
        optimizer.step() # optimizer step -> gradient descent

# plot?
def plot_durations(laptimes):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(laptimes, dtype=torch.float) #duration 대신 다른 것을 y 값으로 -> height !! ****************************
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Maximum Height')
    plt.plot(durations_t.numpy()) #변수 수정
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def get_today():
    now = time.localtime()
    s = "%04d-%02d-%02d_%02d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s

# main, training

'''
main은 따로 빼서, main file을 따로 만들고
통신 코드랑 dqn을 각각 가져와서 조합해서 사용하면 될 듯
'''
def main():
    today = get_today()
    work_dir = "./" + today
    os.makedirs(work_dir + '_' + RACETRACK)

    q = Qnet()
    # q.load_state_dict(torch.load("{}\weigths\model_state_dict_easy1_fin.pt".format(current_dir)))
    q_target = Qnet()

    memory = ReplayBuffer()
    q.load_state_dict(torch.load("{}\\2024-06-15_15-36-36_Oschersleben\\fast-model64.03_771.pt".format(current_dir)))
    q_target.load_state_dict(torch.load("{}\\2024-06-15_15-36-36_Oschersleben\\fast-model64.03_771.pt".format(current_dir)))
    poses = np.array([[0.0702245, 0.3002981, 2.79787]]) # Oschersleben
    #poses = np.array([[0.60070, -0.2753, 1.5707]])  # map easy

    print_interval = 10
    optimizer = optim.Adam(q.parameters(), lr=learning_rate) #----------------------------------------> adam optimizer
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 2000, gamma = 0.9) #Lambda # --------------------> scheduler
    #speed = 3.0 -> ?
    fastlap = 10000.0
    laptimes = []

    for n_epi in range(10000): #10000
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1% #n_epi/200 ------> use exponential?
        obs, r, done, info = env.reset(poses=poses)
        s = q.preprocess_lidar(obs['scans'][0])
        done = False

        #env.render()
        laptime = 0.0

        while not done: # ----------------------------------------> learning starts
            actions = []
            a = q.sample_action(torch.from_numpy(s).float(), epsilon, memory.size())
            action = q.action_to_stearing(a)
            actions.append(action)
            actions = np.array(actions) #actions -> from sample... replay buffer

            obs, r, done, info = env.step(actions) #environment에 action 대입 -> 다음 step의 reward & observation ( = state)가 출력

            s_prime = q.preprocess_lidar(obs['scans'][0]) #next state!
            #print(len(obs['scans'][0])) # added
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100, s_prime, done_mask)) #memory에 ㅔ새롭게 대입
            s = s_prime

            laptime += r
            #env.render(mode='human_fast')

            if done:
                laptimes.append(laptime)
                plot_durations(laptimes)
                lap = round(obs['lap_times'][0], 3)
                if int(obs['lap_counts'][0]) == 2 and fastlap > lap:
                    torch.save(q.state_dict(), work_dir + '_' + RACETRACK + '/fast-model' + str(
                        round(obs['lap_times'][0], 3)) + '_' + str(n_epi) + '.pt')
                    fastlap = lap
                    break

        if memory.size() > train_start: # memory 충분히 차면 with random action
            train(q, q_target, memory, optimizer)#, scheduler) # scheduler added

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%"
                  .format(n_epi, laptime / print_interval, memory.size(), epsilon * 100))

    print('train finish')
    env.close()


def eval():
    env = gym.make('f110_gym:f110-v0',
                   map="{}/maps/{}".format(current_dir, RACETRACK),
                   map_ext=".png", num_agents=1)

    q = Qnet()
    #q.load_state_dict(torch.load("{}\weigths\model_state_dict_easy1_fin.pt".format(current_dir))) # pretrained!
    #poses = np.array([[0.60070, -0.2753, 1.5707]])  # map easy
    #history_1\\2024-06-06_23-24-06_map_easy3\\fast-model27.34_1025.pt
    #\\2024-06-07_01-13-49_Oschersleben\\fast-model78.38_1061.pt # oschersleben
    q.load_state_dict(torch.load("{}\\fastest-model\\fast-model62.97_356.pt".format(current_dir)))
    poses = np.array([[0.0702245, 0.3002981, 2.79787]]) # Oschersleben
    speed = 3.0
    for t in range(5):
        obs, r, done, info = env.reset(poses=poses)
        s = q.preprocess_lidar(obs['scans'][0])

        env.render()
        done = False

        laptime = 0.0

        while not done:
            actions = []

            a = q.action(torch.from_numpy(s).float())
            action = q.action_to_stearing(a)
            actions.append(action)
            actions = np.array(actions)

            obs, r, done, info = env.step(actions)
            s_prime = q.preprocess_lidar(obs['scans'][0])
            #print(obs['poses_y'][0])

            s = s_prime

            laptime += r
            env.render(mode='human_fast')

            if done:
                break
    env.close()


if __name__ == '__main__':
    main()
    #eval()
