import random
random.seed(23)
import numpy as np

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.distributions import Categorical
from plot import plot_results

import matplotlib.pyplot as plt
import pandas as pd


# use_cuda = torch.cuda.is_available()
# device   = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")
torch.autograd.set_detect_anomaly(True)

from ImageEnvironment import GridWorldEnv

from MooreMachine import MooreMachine
from pygame.locals import *

from Classifier import CNN, CNN_feature_extraction
from Visual_DFA_induction import Visual_DFA_induction
from MooreMachine import MinecraftMoore

import seaborn as sns

num_envs = 1
transition_function = {0:{0:2, 1:5, 2:0, 3:1, 4:0}, 1:{0:1, 1:1, 2:1, 3:1, 4:1}, 2:{0:2, 1:3, 2:2, 3:1, 4:2}, 3:{0:3, 1:3, 2:4, 3:1, 4:3}, 4:{0:4, 1:4, 2:4, 3:1, 4:4},
                      5:{0:3, 1:5, 2:5, 3:1, 4:5}}
output_function = [3,4,2,1,0,2]

resize = torchvision.transforms.Resize((128,128))
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    resize,
])

minecraft_machine = MooreMachine(transition_function, output_function)
env = GridWorldEnv(minecraft_machine, "human", False)

termination = False

best_sequences = []
best_rewards = []
best_related_info = []


# MODELS

# CNN feature extractor
cnn = CNN_feature_extraction().to(device)

# ACTOR CRITIC MODEL
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(),
        )
        
    def forward(self, x):

        value = self.critic(x)
        
        probs = self.actor(x)

        dist  = Categorical(probs)

        return dist, value
    
# CONVERT TO THE VECTOR STATE (thanks to the cnn)
def from_obs_to_state(obss):

    obss = torch.tensor(obss.copy()) / 255
    obss = torch.permute(obss, (2,0,1))
    obss = resize(obss)
    
    state = cnn(obss.view(-1, 3, 128, 128))

    return state


# Compute the returns (of the rewards) for one episode
# 
def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        #R = rewards[step] + gamma * R * masks[step]
        m = masks[step].to(device)
        A = rewards[step].to(device)
        B = gamma * R * m
        R = A + B
        returns.insert(0, R)
    return returns

def prepare_dataset(best_sequences, best_rewards, best_related_info, new_sequences, new_rewards, new_info, TT):
    new_sequences = new_sequences[-TT:]
    new_rewards = new_rewards[-TT:]
    new_info = new_info[-TT:]
    for seq, rew, info in zip(new_sequences, new_rewards, new_info):
        if len(best_sequences) == 0:
            best_sequences = new_sequences[:int(TT/2)]
            best_rewards = new_rewards[:int(TT/2)]
            best_related_info = new_info[:int(TT/2)]
            indices = np.argsort(best_rewards)
            best_rewards.sort()
            best_sequences = [best_sequences[i] for i in indices]
            best_related_info = [best_related_info[i] for i in indices]
        else:
            if rew > np.min(best_rewards):
                best_rewards.append(rew)
                best_sequences.append(seq)
                best_related_info.append(info)
                indices = np.argsort(best_rewards)
                best_rewards.sort()
                best_sequences = [best_sequences[i] for i in indices]
                best_related_info = [best_related_info[i] for i in indices]
                best_rewards.pop()
                best_sequences.pop()
                best_related_info.pop()

    last_ten_seq = new_sequences[-10:]
    last_ten_rew = new_rewards[-10:]
    last_ten_info = new_info[-10:]
    return best_sequences, best_rewards, best_related_info, best_sequences+last_ten_seq, best_related_info+last_ten_info


#reinitialize files
f = open("dfa_accuracy.txt", "w")
f.close()

f = open("image_class_accuracy.txt", "w")
f.close()

f = open("Trainrewards.txt", "w")
f.close()

# size of the state vector
num_inputs=11 #was 50

# number of actions
num_outputs=4

#Hyper params:
#hidden_size = 5 #256 hidden size for the Actor Critic
hidden_size = 32 #was 50
lr          = 1e-3

# number of steps per episode
num_steps   = 30

# Initializing the Actor critic model
model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)




# number of episodes (frames)
max_frames   = 1000
frame_idx    = 0
test_rewards = []

# we test the model every TT episodes

TT_grounder=20
TT_policy = 5
# we plot the graph every TTT episode
TTT=10

x_axis = []

# grounding setup
num_states_machine = MinecraftMoore.numb_of_states
num_output_machine = MinecraftMoore.numb_of_rewards
num_symbols_machine = 5


ltl_grounding = Visual_DFA_induction(num_states_machine, num_symbols_machine, num_output_machine, automa_implementation = 'logic_circuit', num_exp=1)
ltl_grounding.deepAutoma.initFromDfa({0:{0:2, 1:5, 2:0, 3:1, 4:0}, 1:{0:1, 1:1, 2:1, 3:1, 4:1}, 2:{0:2, 1:3, 2:2, 3:1, 4:2}, 3:{0:3, 1:3, 2:4, 3:1, 4:3}, 4:{0:4, 1:4, 2:4, 3:1, 4:4},
            5:{0:3, 1:5, 2:5, 3:1, 4:5}}, [3,4,2,1,0,2])
ltl_grounding.deepAutoma.to(device)
ltl_grounding.classifier.to(device)


# we reset the env

obs = env.reset()
obs_ = obs


# first state to start with
obs = torch.tensor(obs.copy()) / 255
obs = torch.permute(obs, (2,0,1))
obs = resize(obs)
state_cnn = cnn((obs.view(-1, 3, 128, 128)).detach())
state_automa = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).to(device)

obs_ = torch.tensor(obs_.copy()) / 255
obs_ = torch.permute(obs_, (2,0,1))
obs_ = resize(obs_)

state = torch.cat((state_cnn, state_automa), dim=-1) #[1,11]

lr=0.0007
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(list(model.parameters()) + list(cnn.parameters()), lr=lr)

advantage_cat = torch.tensor([]).to(device)
log_probs_cat = torch.tensor([]).to(device)

cum_rew_ds = []
image_traj = []
rew_traj = []
info_traj = []
sum_rew_traj = []


all_mean_rewards = []
all_mean_rewards_averaged = []

train_n = 0
while frame_idx < max_frames:

    print("#### echo frame : "+str(frame_idx))
    done = False
    optimizer.zero_grad()
    log_probs = []
    values    = []
    rewards   = []
    masks     = []
    episode_rewards = []
    entropy = 0

    # initializing vectors for the ltl_grounding dataset
    
    curr_traj = []
    curr_rew = []
    curr_info = []
    curr_traj.append(obs_)
    curr_rew.append(-1)
    curr_info.append(3)

    # rollout trajectory
    #print("##echo step : "+str(0))
    while not done:
        state = torch.squeeze(state)
        state.to(device)
        
        dist, value = model(torch.squeeze(state).detach())
        action = dist.sample()
        
        obss, reward, done, info = env.step(action.item())

        obss = torch.tensor(obss.copy()) / 255
        obss = torch.permute(obss, (2,0,1))
        obss = resize(obss)

        curr_traj.append(obss)
        curr_rew.append(reward)
        curr_info.append(info['distance'])

        next_state = obss

        state_grounding = ltl_grounding.classifier(next_state.view(-1, 3, 128, 128))

        next_state_automa, reward_automa = ltl_grounding.deepAutoma.step(state_automa, state_grounding, 1.0)

        state_cnn = cnn((obss.view(-1, 3, 128, 128)).detach())
        state_automa = next_state_automa
        state = torch.cat((state_cnn, state_automa), dim=-1)
    
        # now we store the values
        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        log_prob = torch.unsqueeze(log_prob, 0)
        log_probs.append(log_prob)

        # storing the value retrieved from the Critic
        values.append(value)
        
        # storing the reward retrived from the enbv
        reward = float(reward)
        episode_rewards.append(reward)
        reward = np.expand_dims(reward, axis=0)
        reward = np.expand_dims(reward, axis=0)
        reward = torch.tensor(reward)
        rewards.append(reward)

        formask = 1 if done is True else 0
        formask = np.expand_dims(formask, axis=0)
        formask = np.expand_dims(formask, axis=0)
        formask = torch.tensor(formask)

        masks.append(formask)
   
        image_traj.append(curr_traj)
        rew_traj.append(curr_rew)
        sum_rew_traj.append(np.sum(curr_rew))
        info_traj.append(curr_info)

    frame_idx += 1

    #RESET
    obs = env.reset()

    # first state to start with
    obs = torch.tensor(obs.copy()) / 255
    obs = torch.permute(obs, (2,0,1))
    obs = resize(obs)
    state_cnn = cnn((obs.view(-1, 3, 128, 128)).detach())
    state_automa = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).to(device)
    state = torch.cat((state_cnn, state_automa), dim=-1) #[1,11]
    state = torch.squeeze(state)
    state.to(device)

    # computing the "returns"
    dist, next_value = model(torch.squeeze(state).detach())
    returns = compute_returns(next_value, rewards, masks)
    
    # see source 2 (we use the same loss for updating the weights of both the actor critic and the rnn)
    log_probs = torch.cat(log_probs)
    returns   = torch.cat(returns)
    values    = torch.cat(values)
    log_probs = log_probs.to(device)
    returns = returns.to(device)


    advantage = returns - values

    log_probs_cat = torch.cat((log_probs_cat, log_probs), 0)
    advantage_cat = torch.cat((advantage_cat, advantage), 0)

    torch.cuda.empty_cache()

    if frame_idx % TT_policy == 0:
        if True:
            print("Training policy")
            log_probs_cat = torch.unsqueeze(log_probs_cat, dim=1)
            actor_loss  = -(log_probs_cat * advantage_cat).mean()
            critic_loss = advantage_cat.pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.00001 * entropy
            print("loss: ", loss)
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        train_n += 1

        log_probs_cat = torch.tensor([])
        advantage_cat = torch.tensor([])

    if frame_idx % TT_grounder == 0:
      
        best_sequences, best_rewards, best_related_info, top_seq, top_info = prepare_dataset(best_sequences, best_rewards, best_related_info, image_traj, sum_rew_traj, info_traj, TT_grounder)
        ltl_grounding.set_dataset(image_traj, info_traj)
        ltl_grounding.train_all(16, 100)
        cum_rew_ds = []
        image_traj = []
        rew_traj = []
        info_traj = []
        sum_rew_traj = []

    all_mean_rewards.append(np.sum(np.array(episode_rewards)))
    if(len(all_mean_rewards) >= 10):

            themean = np.mean(all_mean_rewards[-10:])

            all_mean_rewards_averaged.append(themean)

            if frame_idx % TTT == 0:
                #plot rewards
                plt.plot([i for i in range(len(all_mean_rewards_averaged))], all_mean_rewards_averaged)
                plt.xlabel("episode")
                plt.ylabel("mean episode rewards")
                plt.savefig("ImageEnvMeanRewardsReal3"+".png")
                plt.clf()
                plt.close()

                #plot dfa accuracy
                f = open("dfa_accuracy.txt", "r")
                lines = f.readlines()
                f.close()
                accuracies = [float(l) for l in lines]
                plt.plot([TT_grounder*(i+1) for i  in range(len(accuracies))], accuracies)
                plt.xlabel("Episodes")
                plt.ylabel("Sequence classification accuracy")
                plt.savefig("seq_class_accuracy" + ".png")
                plt.clf()
                plt.close()

                #plot symbol grounding accuracy
                f = open("image_class_accuracy.txt", "r")
                lines = f.readlines()
                f.close()
                accuracies = [float(l) for l in lines]
                plt.plot([TT_grounder*(i+1) for i  in range(len(accuracies))], accuracies)
                plt.xlabel("Episode")
                plt.ylabel("Symbol grounding accuracy")
                plt.savefig("img_class_accuracy" + ".png")
                plt.clf()
                plt.close()


    else:
        themean = all_mean_rewards[-1]

    f = open("Trainrewards.txt", "a")
    f.write(str(themean) + "\n")
    f.close()
    print("Mean cumulative reward:", themean)
