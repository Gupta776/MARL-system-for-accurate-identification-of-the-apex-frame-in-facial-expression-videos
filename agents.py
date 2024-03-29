import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pickle
from skimage.feature import local_binary_pattern
from itertools import product


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x):
        return self.model(x)

class LBP_TOP_DDQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon_start, epsilon_end, epsilon_decay):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.update_target_model()
        self.target_network.eval()

        # with open(lbp_top_pickle, 'rb') as f:
        #     self.lbp_top_dict = pickle.load(f)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                return torch.argmax(self.q_network(state)).item()

    # def get_lbp_top_feature(self, frame, index):
    #     print(frame)
    #     frame_path = os.path.join(*frame.split(os.sep)[:-1])
    #     lbp_top_feature = self.lbp_top_dict[frame_path][index]
    #     return lbp_top_feature

    def replay(self, replay_memory, batch_size):
        batch = random.sample(replay_memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).detach()
        max_next_q_values, _ = torch.max(next_q_values, dim=1, keepdim=True)
        target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _compute_lbp_top_feature(self, frame, radius=1, n_points=8):
        lbp = local_binary_pattern(frame, n_points, radius, method='uniform')
        lbp_histogram, _ = np.histogram(lbp.ravel(), bins=np.arange(n_points + 1) + 0.5, density=True)
        lbp_top = lbp_histogram / np.linalg.norm(lbp_histogram)
        return lbp_top

    def get_lbp_top_feature(self, reference_frame, current_frame):
        lbp_top_ref = self._compute_lbp_top_feature(reference_frame)
        lbp_top_cur = self._compute_lbp_top_feature(current_frame)
        lbp_top = np.concatenate((lbp_top_ref, lbp_top_cur))
        return lbp_top
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.model(x)


class A2C:
    def __init__(self, state_dim, action_dim, lr, gamma, entropy_coef):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

    def select_action(self, state):
        with torch.no_grad():
            probs = self.actor(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action.item(), log_prob, entropy

    def compute_advantages(self, rewards, values, next_value):
        advantages = torch.zeros_like(rewards)
        returns = next_value
        for t in reversed(range(len(rewards))):
            returns = rewards[t] + self.gamma * returns
            advantages[t] = returns - values[t]

        return advantages

    def train(self, states, log_probs, rewards, entropies, next_state):
        values = self.critic(states)
        next_value = self.critic(next_state).detach()
        advantages = self.compute_advantages(rewards, values, next_value)

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        loss = actor_loss + critic_loss - self.entropy_coef * entropies.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def _compute_lbp_top_feature(self, frame, radius=1, n_points=8):
        lbp = local_binary_pattern(frame, n_points, radius, method='uniform')
        lbp_histogram, _ = np.histogram(lbp.ravel(), bins=np.arange(n_points + 1) + 0.5, density=True)
        lbp_top = lbp_histogram / np.linalg.norm(lbp_histogram)
        return lbp_top

    def get_lbp_top_feature(self, reference_frame, current_frame):
        lbp_top_ref = self._compute_lbp_top_feature(reference_frame)
        lbp_top_cur = self._compute_lbp_top_feature(current_frame)
        lbp_top = np.concatenate((lbp_top_ref, lbp_top_cur))
        return lbp_top
    
class DQN(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.action_dim = action_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.conv(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

def preprocess_state(state):
    state = np.array(state, dtype=np.float32) / 255.0
    state = np.expand_dims(state, 0)  # Add a channel dimension for grayscale image
    #state = np.transpose(state, (2, 0, 1))
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    return state

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon_start, epsilon_end, epsilon_decay):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)


    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = preprocess_state(state)
        q_values = self.q_net(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack([preprocess_state(state).squeeze(0) for state in states])
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack([preprocess_state(next_state).squeeze(0) for next_state in next_states])
        dones = torch.FloatTensor(dones)

        q_values = self.q_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_end)

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)
