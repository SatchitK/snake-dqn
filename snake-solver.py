import pygame
import random
import numpy as np
import torch
from collections import deque, namedtuple
import time
import torch.nn as nn
import torch.optim as optim

WIDTH, HEIGHT = 640, 640
BLOCK_SIZE = 20
SPEED = 100

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        self.head = [WIDTH // 2, HEIGHT // 2]
        self.snake = [self.head[:], [self.head[0] - BLOCK_SIZE, self.head[1]], [self.head[0] - 2 * BLOCK_SIZE, self.head[1]]]
        self.spawn_food()
        self.score = 0
        self.frame = 0
        self.done = False
        return self.get_state()

    def spawn_food(self):
        while True:
            x = random.randrange(0, WIDTH, BLOCK_SIZE)
            y = random.randrange(0, HEIGHT, BLOCK_SIZE)
            if [x, y] not in self.snake:
                self.food = [x, y]
                break

    def step(self, action):
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        idx = directions.index(self.direction)
        if action == 1:
            idx = (idx + 1) % 4
        elif action == 2:
            idx = (idx - 1) % 4
        self.direction = directions[idx]
        x, y = self.head
        if self.direction == 'UP':
            y -= BLOCK_SIZE
        elif self.direction == 'DOWN':
            y += BLOCK_SIZE
        elif self.direction == 'LEFT':
            x -= BLOCK_SIZE
        elif self.direction == 'RIGHT':
            x += BLOCK_SIZE
        self.head = [x, y]
        self.snake.insert(0, self.head[:])
        reward = 0
        self.done = False
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.spawn_food()
        else:
            self.snake.pop()
            reward = 0.1  
        if (x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT or self.head in self.snake[1:]):
            self.done = True
            reward = -10
        self.frame += 1
        return self.get_state(), reward, self.done, {}

    def get_state(self):
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        idx = directions.index(self.direction)
        left_dir = directions[(idx - 1) % 4]
        right_dir = directions[(idx + 1) % 4]
        def danger(dir):
            x, y = self.head
            if dir == 'UP':
                y -= BLOCK_SIZE
            elif dir == 'DOWN':
                y += BLOCK_SIZE
            elif dir == 'LEFT':
                x -= BLOCK_SIZE
            elif dir == 'RIGHT':
                x += BLOCK_SIZE
            if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT or [x, y] in self.snake:
                return 1
            return 0
        food_left = int(self.food[0] < self.head[0])
        food_right = int(self.food[0] > self.head[0])
        food_up = int(self.food[1] < self.head[1])
        food_down = int(self.food[1] > self.head[1])
        state = [
            danger(self.direction),
            danger(right_dir),
            danger(left_dir),
            food_left,
            food_right,
            food_up,
            food_down,
            idx / 4.0
        ]
        return np.array(state, dtype=np.float32)

    def render(self, screen):
        screen.fill((0, 0, 0))
        for s in self.snake:
            pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(s[0], s[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class Agent:
    def __init__(self, state_dim, action_dim):
        self.memory = deque(maxlen=10000)
        self.model = DQN(state_dim, action_dim)
        self.target = DQN(state_dim, action_dim)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = 0.9
        self.batch_size = 128
        self.steps = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.total_reward_history = []
        self.avg_length_history = []
        self.max_score = 0

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_vals = self.model(state_tensor)
            return int(torch.argmax(q_vals).item())

    def store(self, *args):
        self.memory.append(Transition(*args))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*batch))
        state = torch.tensor(np.array(batch.state), dtype=torch.float32)
        action = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1)
        reward = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
        next_state = torch.tensor(np.array(batch.next_state), dtype=torch.float32)
        done = torch.tensor([float(r == -10) for r in batch.reward], dtype=torch.float32).unsqueeze(1)
        q_values = self.model(state).gather(1, action)
        next_q = self.target(next_state).max(1)[0].detach().unsqueeze(1)
        expected_q = reward + self.gamma * next_q * (1 - done)
        loss = nn.MSELoss()(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.steps % 100 == 0:
            self.target.load_state_dict(self.model.state_dict())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class TrainingVisualizer:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake DQN Training Visualization")
        self.font = pygame.font.SysFont("Arial", 16)
        self.large_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.game_surface = pygame.Surface((WIDTH, HEIGHT))
        self.bg_color = (30, 30, 40)
        self.text_color = (220, 220, 220)
        self.graph_bg = (50, 50, 60)
        self.reward_color = (0, 200, 100)
        self.length_color = (100, 100, 255)

    def render_game(self, game):
        game.render(self.game_surface)
        self.screen.blit(self.game_surface, (20, 80))
        pygame.draw.rect(self.screen, (200, 200, 200), (20, 80, WIDTH, HEIGHT), 2)

    def render_stats(self, agent, episode, game):
        title = self.large_font.render(f"Snake DQN Training - Episode {episode}", True, self.text_color)
        self.screen.blit(title, (20, 20))
        stats = [
            f"Score: {game.score}",
            f"Steps: {game.frame}",
            f"Epsilon: {agent.epsilon:.4f}",
            f"Memory Size: {len(agent.memory)}",
            f"Max Score: {agent.max_score}"
        ]
        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, self.text_color)
            self.screen.blit(text, (WIDTH + 50, 80 + i * 25))
        if agent.total_reward_history:
            self.draw_graph(agent.total_reward_history[-100:], WIDTH + 50, 220, 300, 150, "Rewards", self.reward_color)
        if agent.avg_length_history:
            self.draw_graph(agent.avg_length_history[-100:], WIDTH + 50, 400, 300, 150, "Avg Snake Length", self.length_color)

    def draw_graph(self, data, x, y, width, height, title, color):
        pygame.draw.rect(self.screen, self.graph_bg, (x, y, width, height))
        title_text = self.font.render(title, True, self.text_color)
        self.screen.blit(title_text, (x + 10, y + 5))
        pygame.draw.line(self.screen, self.text_color, (x, y + height - 20), (x + width - 10, y + height - 20), 1)
        pygame.draw.line(self.screen, self.text_color, (x + 10, y + 20), (x + 10, y + height - 10), 1)
        if len(data) > 1:
            max_val = max(max(data), 1)
            min_val = min(min(data), 0)
            data_range = max_val - min_val or 1
            points = []
            for i, val in enumerate(data):
                px = x + 10 + (i / (len(data) - 1)) * (width - 20)
                py = y + height - 20 - ((val - min_val) / data_range) * (height - 40)
                points.append((px, py))
            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)

def train_agent_with_visualization(episodes=200):
    visualizer = TrainingVisualizer(width=800, height=600)
    game = SnakeGame()
    agent = Agent(state_dim=8, action_dim=3)
    clock = pygame.time.Clock()
    running = True
    MAX_STEPS_PER_EPISODE = 1000
    for ep in range(episodes):
        state = game.reset()
        total_reward = 0
        episode_steps = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    return agent
            action = agent.select_action(state)
            next_state, reward, done, _ = game.step(action)
            agent.store(state, action, next_state, reward)
            agent.train()
            state = next_state
            total_reward += reward
            agent.steps += 1
            episode_steps += 1
            visualizer.screen.fill(visualizer.bg_color)
            visualizer.render_game(game)
            visualizer.render_stats(agent, ep+1, game)
            pygame.display.flip()
            speed = 30 if game.score > 5 else 15
            clock.tick(speed)
            if done or episode_steps >= MAX_STEPS_PER_EPISODE:
                if episode_steps >= MAX_STEPS_PER_EPISODE and not done:
                    print(f"Episode {ep+1} reached time limit of {MAX_STEPS_PER_EPISODE} steps")
                break
        agent.total_reward_history.append(total_reward)
        agent.avg_length_history.append(game.score + 3)
        if game.score > agent.max_score:
            agent.max_score = game.score
        print(f"Episode {ep+1}/{episodes}, Score: {game.score}, Reward: {total_reward:.1f}, Epsilon: {agent.epsilon:.3f}, Steps: {episode_steps}")
        pygame.time.delay(500)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return agent
    pygame.quit()
    return agent

def play_with_agent():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    game = SnakeGame()
    agent = Agent(state_dim=8, action_dim=3)
    agent.model.load_state_dict(torch.load("snake_dqn.pth"))
    agent.epsilon = 0
    state = game.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        action = agent.select_action(state)
        state, _, done, _ = game.step(action)
        game.render(screen)
        font = pygame.font.SysFont("Arial", 18)
        score_text = font.render(f"Score: {game.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
        pygame.display.flip()
        clock.tick(60)
        if done:
            state = game.reset()

if __name__ == "__main__":
    play_with_agent()