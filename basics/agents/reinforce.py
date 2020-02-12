import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from basics.agents.base_agent import Agent

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, observation_space, action_space_size):
        super(Policy, self).__init__()

        if len(observation_space) != 1:
            raise RuntimeError("The Policy specified by this agent only supports 1d obs space")

        self.affine1 = nn.Linear(observation_space[0], 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, action_space_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

class ReinforceAgent(Agent):

    def __init__(self, args, observation_space, action_space_size):
        self.policy = Policy(observation_space, action_space_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma = args.gamma
        self.episode_reward = 0
        self.running_reward = 10
        self.writer = SummaryWriter()
        
    
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def finish_iteration(self, reward, iteration_count):
        self.policy.rewards.append(reward)
        self.episode_reward += reward

    def finish_episode(self, episode):
        self.running_reward = 0.05 * self.episode_reward + (1 - 0.05) * self.running_reward
        R = 0
        policy_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]
        self.writer.add_scalar("Last reward", self.episode_reward, episode)
        self.writer.add_scalar("Average reward", self.running_reward, episode)
        self.episode_reward = 0
    
    def check_solved_env(self, reward_threshold):
        return self.running_reward > reward_threshold
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
        return parser