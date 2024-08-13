import chess
import numpy as np
import gym
from gym import spaces

class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        
        # Action space: 64 * 64 = 4096 possible moves (from any square to any square)
        self.action_space = spaces.Discrete(64 * 64)
        
        # Observation space: 8x8 board with 12 possible pieces (6 white, 6 black) + empty
        self.observation_space = spaces.Box(low=0, high=12, shape=(8, 8), dtype=np.uint8)

    def reset(self):
        self.board = chess.Board()
        return self._get_observation()

    def step(self, action):
        from_square = action // 64
        to_square = action % 64
        
        move = chess.Move(from_square, to_square)
        
        if move in self.board.legal_moves:
            self.board.push(move)
            done = self.board.is_game_over()
            reward = self._get_reward()
        else:
            done = True
            reward = -1  # Penalty for illegal move

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        print(self.board)

    def _get_observation(self):
        obs = np.zeros((8, 8), dtype=np.uint8)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                color = int(piece.color)
                piece_type = piece.piece_type - 1
                obs[i // 8][i % 8] = piece_type + 6 * color + 1
        return obs

    def _get_reward(self):
        if self.board.is_checkmate():
            return 1 if self.board.turn == chess.BLACK else -1
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        else:
            return 0  # No reward for intermediate states

# Example usage
env = ChessEnv()

# Reset the environment
obs = env.reset()
env.render()

# Play a few random moves
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# First, include the ChessEnv class from the previous response

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 4096)  # 64 * 64 possible moves

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0).to(self.device)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state))
            target_f = self.model(state)
            target_f[0][action] = target
            loss = self.criterion(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_chess_ai():
    env = ChessEnv()
    state_size = (8, 8)
    action_size = 4096
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    n_episodes = 10000

    for e in range(n_episodes):
        state = env.reset()
        for time in range(500):  # limit the maximum number of moves
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{n_episodes}, score: {reward}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 2000 == 0:
            torch.save(agent.model.state_dict(), f"chess_model_episode_{e}.pth")

# Run the training
train_chess_ai()