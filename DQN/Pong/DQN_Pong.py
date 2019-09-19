from dataclasses import dataclass, field
from PIL import Image
from torch.nn.init import kaiming_normal_
from typing import Any, List
from utils.atari_wrappers import MaxAndSkipEnv, NoopResetEnv 
import argparse
import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import model
import numpy as np
import pandas as pd
import random
import time
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim

# in the following, number of frames are counted without emulator's initialization frames so that the number corresponds to number of actions executed by the agent
parser = argparse.ArgumentParser(description='DQN algorithm - Atari Pong')
parser.add_argument('--batch_size', type=int, default=32, help='batch size (default: 32)')
parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor gamma used in the Q-learning update (default: 0.99)')
parser.add_argument('--eps_eval', type=float, default=0.05, help='eps value used in eps-greedy exploration during evaluation (default: 0.05)')
parser.add_argument('--eps_train_begin', type=float, default=1, help='initial value of eps in eps-greedy exploration during training (default: 1)')
parser.add_argument('--eps_train_end', type=float, default=0.1, help='final value of eps in eps-greedy exploration during training (default: 0.1)')
parser.add_argument('--eps_valid', type=float, default=0.05, help='eps value used in eps-greedy exploration during validation (default: 0.05)')
parser.add_argument('--game', type=str, default='pong', help='game (default: pong)')
parser.add_argument('--load_model', action='store_true', help='load a pretrained model (default: False)')
parser.add_argument('--lr', type=float, default=0.00025, help='learning rate (default: 0.00025)')
parser.add_argument('--mode', type=str, default='eval', help="choose 'train' to train, 'eval' to evaluate, or 'traineval' to both train and evaluate an agent. (default: 'eval')")
parser.add_argument('--n_episodes_eval', type=int, default=30, help="number of episodes over which to evaluate the agent's performance. (default: 30)")
parser.add_argument('--n_frames_eps', type=int, default=1000000, help='number of frames over which eps in eps-greedy exploration is annealed during training (default: 1 M)')
parser.add_argument('--n_frames_eval', type=int, default=18000, help='max number of frames that a single episode can last in evaluation (default: 18 k, which is 5 min)')
parser.add_argument('--n_frames_gif', type=int, default=1000, help='number of last frames of a random evaluation episode that are saved in gif. if zero, all the frames are saved. (default: 1 k)')
parser.add_argument('--n_frames_input', type=int, default=4, help='number of most recent frames from emulator stacked as an input to Q-network (default: 4)')
parser.add_argument('--n_frames_total', type=int, default=20000000, help="number of training frames in total (default: 50 M)")
parser.add_argument('--n_frames_train', type=int, default=250000, help='number of training frames in an epoch (default: 250 k)')
parser.add_argument('--n_frames_valid', type=int, default=125000, help='number of validation frames in an epoch (default: 125 k)')
parser.add_argument('--replaymemory_capacity_full', type=int, default=1000000, help="size of the replay memory, which stores most recent frames (default: 1 M)")
parser.add_argument('--replaymemory_capacity_start', type=int, default=50000, help='size of the replay memory that is populated using random actions before training begins (default: 50 k)')
parser.add_argument('--save_gif', action='store_true', help='save a gif of the best Q-network playing a random episode during evaluation (default: False)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--update_frequency_Q_net', type=int, default=4, help="number of actions executed by the agent between back-to-back Q-network's SGD updates (default: 4)")
parser.add_argument('--update_frequency_target', type=int, default=10000, help="number of actions executed by the agent between back-to-back target network's updates. this correponds to the DeepMind's implementation, but not to the paper, which claims that they use number of parameter updates instead of number of frames (default: 10 k)")
args = parser.parse_args()

batch_size = args.batch_size
discount_factor = args.discount_factor
eps_eval = args.eps_eval
eps_train_begin = args.eps_train_begin
eps_train_end = args.eps_train_end
eps_valid = args.eps_valid
game = args.game
load_model = args.load_model
lr = args.lr
mode = args.mode
n_episodes_eval = args.n_episodes_eval
n_frames_eps = args.n_frames_eps
n_frames_eval = args.n_frames_eval
n_frames_gif = args.n_frames_gif
n_frames_input = args.n_frames_input
n_frames_total = args.n_frames_total
n_frames_train = args.n_frames_train
n_frames_valid = args.n_frames_valid
replaymemory_capacity_full = args.replaymemory_capacity_full
replaymemory_capacity_start = args.replaymemory_capacity_start
save_gif = args.save_gif
seed = args.seed
update_frequency_Q_net = args.update_frequency_Q_net
update_frequency_target = args.update_frequency_target

n_frames_epoch = {'init': replaymemory_capacity_start, 'train': n_frames_train, 'valid': n_frames_valid}

# set paths
path_model = 'models/Q_net_best.pt' # for loading a pretrained model
path_best_model_new = 'models/Q_net_best_new.pt' # for saving the best model and if 'traineval' mode is used, for loading the best newly trained model also
path_gif = 'visualizations/random_episode_eval.gif'
path_statistics_dir = 'statistics/'

plt.ion()

FRAME_SIZE_PREPROCESSED = (84, 84) # preprocessed frame size for Atari games

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set seeds to RNGs
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if device.type == 'cuda':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if game == 'pong':
    game_env = 'PongNoFrameskip-v4'
else:
    raise ValueError("Game name not recognized, for now use only 'pong' as the game.")

def make_env(env_name):
    env = gym.make(env_name)
    env = NoopResetEnv(env, seed)
    env = MaxAndSkipEnv(env)
    return env

env = make_env(game_env)

action_space = env.action_space
n_actions = action_space.n

# model related
def init_weights(m):
    layer = type(m)
    if layer == nn.Conv2d:
        kaiming_normal_(m.weight, nonlinearity='relu')
    elif layer == nn.Linear:
        if m.out_features == 512:
            kaiming_normal_(m.weight, nonlinearity='relu')

# line 2 (1/2): initialize Q-network with random weights
Q_net = model.Q_net_Pong(FRAME_SIZE_PREPROCESSED, n_frames_input, n_actions)
if mode == 'eval':
    Q_net.load_state_dict(torch.load(path_model))
    Q_net = Q_net.to(device)
else:
    if load_model == True:
        Q_net.load_state_dict(torch.load(path_model))
        print("using a pretrained model")
    else:
        # line 2 (2/2): use Kaiming normal initialization for all the relu layers in the Q-network
        Q_net.apply(init_weights)
    
    # line 3: initialize target Q-network by copying the weights of the main Q-network to it
    Q_net_target = model.Q_net_Pong(FRAME_SIZE_PREPROCESSED, n_frames_input, n_actions)
    Q_net_target.load_state_dict(Q_net.state_dict())
    Q_net_target.eval()

    Q_net, Q_net_target = Q_net.to(device), Q_net_target.to(device)
    
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(Q_net.parameters(), lr=1e-5)

def stack_initial_observations():
    observation_current = env.reset()
    observations_current = observation_current[np.newaxis, :]
    env.render()
    for _ in range(n_frames_input-1):
        observations_current = np.vstack((observation_current[np.newaxis, :], observations_current))
    return observations_current

resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize(FRAME_SIZE_PREPROCESSED, interpolation=Image.BILINEAR),
                    T.ToTensor()])
#topilimg = T.ToPILImage()

def preprocess(observations):
    n_frames = observations.shape[0]
    observations_preprocessed = torch.zeros(n_frames, FRAME_SIZE_PREPROCESSED[0], FRAME_SIZE_PREPROCESSED[1])
    for i_observation in range(n_frames):
        observation_preprocessed = 255*resize(observations[i_observation])
        #img = topilimg(observation_preprocessed)
        #img.show()
        #time.sleep(3)
        observations_preprocessed[i_observation] = observation_preprocessed
    return observations_preprocessed.numpy().astype('uint8')

# gives the initial frame stack to start a new episode
def return_initial_preprocessed_frame_stack():
    observations_current = stack_initial_observations() # line 5 (1/2): for each new episode, initialize frame stack
    observations_current = preprocess(observations_current) # line 5 (2/2): preprocess the frame stack
    return observations_current

def select_action(observations_current, frame_cnt_train=None, eps=None):
    if eps == None:
        if frame_cnt_train >= n_frames_eps:
            eps = eps_train_end
        else:
            eps = eps_train_begin + (eps_train_end - eps_train_begin) * (frame_cnt_train/n_frames_eps)
    # lines 7 and 8: eps-greedy strategy
    if random.random() > eps: # line 8: exploit learned information by choosing the best action according to the Q-network
        with torch.no_grad():
            action = Q_net(torch.from_numpy(observations_current.astype('float32') / 255).float().to(device)).max(1)[1].item()
            return action
    else: # line 7: explore options by choosing a random action
        return action_space.sample()

@dataclass
class Transition:
    observations_current: Any
    observations_next: Any
    action: int
    reward: int
    terminal: bool
    
    def __iter__(self):
        return iter([self.observations_current, self.observations_next, self.action, self.reward, self.terminal])

@dataclass
class ReplayMemory:
    capacity: int
    memory: List[Transition] = field(default_factory=list)
    position: int = 0

    def store(self, transition):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

replaymemory = ReplayMemory(replaymemory_capacity_full) # line 1: initialize replay memory to a given capacity

def create_transition(observations_current, observation_next, action, reward, terminal):
    observation_next = observation_next[np.newaxis, :]
    observation_next = preprocess(observation_next)
    observations_next = np.vstack((observation_next, observations_current[:-1]))
    return Transition(observations_current, observations_next, action, reward, terminal)

def optimize_model(batch_size):
    n_transitions = len(replaymemory)
    if n_transitions < batch_size:
        batch_size = n_transitions
    
    batch_transitions = replaymemory.sample(batch_size) # line 12: sample a random batch of transitions
    batch = Transition(*zip(*batch_transitions)) # transition of batches
    batch_observations_current = torch.from_numpy(np.asarray(batch.observations_current).astype('float32') / 255)
    batch_action = torch.tensor(batch.action).view(-1, 1).to(device)
    batch_reward = torch.tensor(batch.reward).view(-1, 1).to(device)
    batch_terminal = torch.tensor(batch.terminal)

    Q_values_next = torch.zeros(batch_size, 1).to(device)
    nonterminal_indexes = (batch_terminal == 0).nonzero().view(-1)
    batch_nonterminal = filter(lambda transition: transition.terminal == False, batch_transitions)
    observations_next_nonterminal = torch.from_numpy(np.asarray(list(map(lambda transition: transition.observations_next, batch_nonterminal))).astype('float32') / 255)
    Q_values_next[nonterminal_indexes] = Q_net_target(observations_next_nonterminal.float().to(device)).max(1)[0].view(-1, 1).detach()
    target = batch_reward + discount_factor * Q_values_next # line 13: compute the approximated target values
    
    Q_values_current = Q_net(batch_observations_current.float().to(device)).gather(1, batch_action)
    
    loss = criterion(Q_values_current, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # line 14: update the weights of the Q-network

class Results(object):
    def __init__(self):
        self.episode = 0
        self.epoch = 0
        self.reward_episode = 0
        self.results = []
        self.n_epochs = n_frames_total // n_frames_train

    def add_episode_reward(self, phase):
        self.results.append({'phase':phase, 'epoch':self.epoch, 'episode':self.episode, 'reward':self.reward_episode})
        self.episode += 1
        self.reward_episode = 0
        
    def compute_avg_and_sd_reward_per_episode(self, phase):
        results_phase_epoch = list(filter(lambda result_episode: result_episode['epoch'] == self.epoch and result_episode['phase'] == phase, self.results))
        rewards_phase_epoch = [result_episode['reward'] for result_episode in results_phase_epoch]
        average_reward_epoch, sd_reward_epoch = np.mean(rewards_phase_epoch), np.std(rewards_phase_epoch)
        self.results.append({'phase':phase, 'epoch':self.epoch, 'average_reward':average_reward_epoch, 'sd_reward':sd_reward_epoch})
        if phase == 'valid':
            self.epoch += 1
        return average_reward_epoch, sd_reward_epoch

    def plot_average_reward_per_episode(self):
        results_valid = list(filter(lambda result_episode: 'average_reward' in result_episode and result_episode['phase'] == 'valid', self.results))
        average_rewards = [result_epoch['average_reward'] for result_epoch in results_valid]
        plt.figure(0)
        plt.clf()
        plt.title('Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Average reward per episode')
        plt.plot(average_rewards, '-o')
        plt.pause(0.1)

    def print_average_reward_of_past_100_episodes(self, phase, frame_cnt):
        results_phase = list(filter(lambda result_episode: result_episode['phase'] == phase and 'reward' in result_episode, self.results))
        results_phase_past_100_episodes = results_phase[-100:]
        average_reward_past_100_episodes = sum(result_episode['reward'] for result_episode in results_phase_past_100_episodes) / len(results_phase_past_100_episodes)
        if phase not in ('eval', 'init'):
            print('epoch: {}/{} phase: {} frame: {}/{} avg. reward: {:.2f}'.format((self.epoch+1), self.n_epochs, phase, frame_cnt, n_frames_epoch[phase], average_reward_past_100_episodes))
        elif phase == 'eval':
            print('episode: {}/{} avg. reward: {:.2f}'.format(self.episode, n_episodes_eval, average_reward_past_100_episodes))
        else:
            print('phase: {} frame: {}/{} avg. reward: {:.2f}'.format(phase, frame_cnt, n_frames_epoch[phase], average_reward_past_100_episodes))

    def save_results(self, phase):
        results_phase = list(filter(lambda result_episode: result_episode['phase'] == phase, self.results))
        results_df = pd.DataFrame.from_records(results_phase, index='epoch')
        results_df = results_df[['episode', 'reward', 'average_reward', 'sd_reward']]
        results_df.to_csv(path_statistics_dir+str(phase)+'_new.csv', sep=' ')

def process_results(phase, results, frame_cnt):
    results.add_episode_reward(phase)
    results.print_average_reward_of_past_100_episodes(phase, frame_cnt)

def train():
    print("train an agent")
    replaymemory_ram = (replaymemory_capacity_full * 2 * n_frames_input * FRAME_SIZE_PREPROCESSED[0] * FRAME_SIZE_PREPROCESSED[1]) / (1024 ** 3)
    if replaymemory_ram >= 4:
        print('WARNING: replay memory will take approximately {:.0f} GB of RAM'.format(replaymemory_ram))
    
    print('begin initialization')
    phase = 'init'
    Q_net.eval()
    
    observations_current = return_initial_preprocessed_frame_stack()
    frame_cnt = {'init':0, 'train':0, 'valid':0} # emulator's initialization frames will not be counted so that the number corresponds to number of actions executed by the agent
    
    results = Results()
    # Deep Q-learning with experience replay algorithm    
    while results.epoch < results.n_epochs:
        if phase == 'train':
            Q_net.eval()
            frame_cnt_train_total = results.epoch * n_frames_train + frame_cnt['train']
            action = select_action(observations_current, frame_cnt_train=frame_cnt_train_total) # lines 7 and 8 are commented in the function
            Q_net.train()
        elif phase == 'valid':
            action = select_action(observations_current, eps=eps_valid)
        else:
            action = action_space.sample()
        
        observation_next, reward, terminal, _, _ = env.step(action) # line 9: execute action, which advances environment by k=4 frames
        results.reward_episode += reward
        frame_cnt[phase] += 1
        env.render()

        # v line 10: create a new preprocessed frame stack by adding the new frame to the current stack and popping out the oldest frame
        transition = create_transition(observations_current, observation_next, action, reward, terminal)
        if phase != 'valid':
            replaymemory.store(transition) # line 11

        if phase == 'train':
            if frame_cnt['train'] % update_frequency_Q_net == 1:
                optimize_model(batch_size) # lines 12-14 are commented in the function
            
            if frame_cnt['train'] % update_frequency_target == 1:
                Q_net_target.load_state_dict(Q_net.state_dict()) # line 15: update the target network                

            if frame_cnt['train'] % n_frames_train == 0:
                if terminal:
                    process_results(phase, results, frame_cnt[phase])
                else:
                    results.reward_episode = 0
                _, _ = results.compute_avg_and_sd_reward_per_episode(phase)           
                results.save_results(phase)
                frame_cnt[phase] = 0
                print('end training and begin validation for this epoch')
                phase = 'valid'
                Q_net.eval()
                observations_current = return_initial_preprocessed_frame_stack()
                continue

        else:
            if (phase == 'valid' and frame_cnt[phase] == n_frames_valid) or (phase == 'init' and frame_cnt[phase] == replaymemory_capacity_start):
                if terminal:
                    process_results(phase, results, frame_cnt[phase])
                else:
                    results.reward_episode = 0
                
                if phase == 'valid':                    
                    average_reward_current, _ = results.compute_avg_and_sd_reward_per_episode(phase)
                    try:
                        if average_reward_current > average_reward_best:
                            print("NEW BEST MODEL!")
                            torch.save(Q_net.state_dict(), path_best_model_new)
                            average_reward_best = average_reward_current
                    except NameError:
                        torch.save(Q_net.state_dict(), path_best_model_new)
                        average_reward_best = average_reward_current
                    results.plot_average_reward_per_episode()
                    results.save_results(phase)
                    frame_cnt[phase] = 0
                    if results.epoch < results.n_epochs:
                        print('end validation and begin a new epoch with training')
                    else:
                        print('end validation and stop training')
                else:
                    print('end initialization and begin an epoch with training')
                
                phase = 'train'
                Q_net.train()
                observations_current = return_initial_preprocessed_frame_stack()
                continue

        if terminal:
            process_results(phase, results, frame_cnt[phase])
            observations_current = return_initial_preprocessed_frame_stack()
            continue

        observations_current = transition.observations_next

    if mode == 'train':
        plt.ioff()
        plt.show()

def evaluate():
    print("begin evaluation")
    phase = 'eval'
    
    if save_gif:
        print('WARNING: saving a gif requires relatively high amount of RAM, around 5 GB with the default setting of 1 k frames')
        
    # for animation
    images = []
    episode_animation = np.random.randint(n_episodes_eval)
    episode_animation = 0

    results = Results()
    
    while results.episode < n_episodes_eval:
        frame_cnt_episode = 0
        observations_current = return_initial_preprocessed_frame_stack()
        
        while frame_cnt_episode <= n_frames_eval:
            action = select_action(observations_current, eps=eps_eval)            
            observation_next, reward, terminal, _, all_frames = env.step(action)
            results.reward_episode += reward
            frame_cnt_episode += 1
            env.render()
            
            transition = create_transition(observations_current, observation_next, action, reward, terminal)
            
            if results.episode == episode_animation:
                images += all_frames

            if terminal:
                process_results(phase, results, frame_cnt_episode)
                break
  
            observations_current = transition.observations_next

    average_reward, sd_reward = results.compute_avg_and_sd_reward_per_episode(phase)
    print('evaluation reward: average: {:.1f}, standard deviation: {:.1f}'.format(average_reward, sd_reward))

    results.save_results(phase)

    fig1 = plt.figure(figsize=(4,4), tight_layout=True)   
    plt.axis("off")
    ims = [[plt.imshow(image, animated=True)] for image in images]
    interval_experimental = (1/5)*(1000/(60/4)) # experimental factor * 1000 ms / (60/4) frames
    interval = max(20, interval_experimental)
    ani = animation.ArtistAnimation(fig1, ims, interval=interval, repeat_delay=interval, blit=True)

    if save_gif:
        ani_duration = 90
        print('showing the animation of a random episode for {} seconds before saving the last {} frames of it in a gif'.format(ani_duration, n_frames_gif))
        plt.pause(ani_duration)
        print('saving the gif now, takes a few minutes')
        fig2 = plt.figure(figsize=(4,4), tight_layout=True)
        plt.axis("off")
        ims = [[plt.imshow(image, animated=True)] for image in images[-n_frames_gif:]]
        ani = animation.ArtistAnimation(fig2, ims, interval=interval, repeat_delay=interval, blit=True)
        ani.save(path_gif, writer='imagemagick')
        print("gif saved")

    plt.ioff()
    plt.show()

def main():
    if mode == 'eval':
        evaluate()
    elif mode == 'traineval':
        train()
        Q_net.load_state_dict(torch.load(path_best_model_new))
        evaluate()
    elif mode == 'train':
        train()
    else:
        raise ValueError("mode is invalid, choose a valid mode: 'eval', 'traineval', or 'train'")

main()