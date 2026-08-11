import gymnasium as gym
import torch
import torch.nn as nn
import random as rand
from network import Network

#initializing Networks
target_network = Network()
online_network = Network()

#in the begining the two networks should be equal
target_network.load_state_dict(online_network.state_dict())

env = gym.make("Blackjack-v1")
gamma = 1
epsilon = 0.1

optimizer = torch.optim.Adam(online_network.parameters(), lr= 0.001)
loss_function = nn.MSELoss()

# e-greedy action chooser
def choose_action(state):
    if rand.random() < epsilon:
        return rand.choice([0, 1])

    state_tensor = encode_state(state)

    with torch.no_grad():
        q_values = online_network(state_tensor)

    return q_values.argmax().item()


def encode_state(state):
    player_sum, dealer_card, usable_ace = state

    return torch.tensor(
        [
            player_sum / 21,
            dealer_card / 10,
            float(usable_ace)
            ], dtype=torch.float32
        )


def train():
    num_episodes = 100000
    target_update_frequency = 100

    episode_rewards = []

    for episode in range(num_episodes):

        state, _ = env.reset()

        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):

            action = choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            # Current Q-value
            state_tensor = encode_state(state)
            q_values = online_network(state_tensor)
            q_value = q_values[action]

            # Target
            with torch.no_grad():

                if terminated or truncated:
                    target = torch.tensor(reward, dtype=torch.float32)

                else:
                    next_state_tensor = encode_state(next_state)
                    next_q_values = target_network(next_state_tensor)
                    max_next_q_value = next_q_values.max()

                    target = reward + gamma * max_next_q_value

            # Loss
            loss = loss_function(q_value, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        episode_rewards.append(total_reward)

        if (episode + 1) % target_update_frequency == 0:
            target_network.load_state_dict(online_network.state_dict())

        if (episode + 1) % 1000 == 0:

            avg_reward = sum(episode_rewards[-1000:]) / 1000

            print(
                f"Episode: {episode + 1}, "
                f"Average reward: {avg_reward:.3f}"
            )


def evaluate(num_episodes=10000):

    wins = 0
    losses = 0
    draws = 0

    total_reward = 0

    for _ in range(num_episodes):

        state, _ = env.reset()

        terminated = False
        truncated = False

        while not (terminated or truncated):

            state_tensor = encode_state(state)

            with torch.no_grad():
                q_values = online_network(state_tensor)

            # Pure exploitation
            action = q_values.argmax().item()

            state, reward, terminated, truncated, _ = env.step(action)

        # Episode is finished
        total_reward += reward

        if reward == 1:
            wins += 1

        elif reward == -1:
            losses += 1

        else:
            draws += 1

    win_rate = wins / num_episodes
    loss_rate = losses / num_episodes
    draw_rate = draws / num_episodes

    average_reward = total_reward / num_episodes

    print(f"Games:          {num_episodes}")
    print(f"Wins:           {wins}")
    print(f"Losses:         {losses}")
    print(f"Draws:          {draws}")
    print(f"Win rate:       {win_rate:.2%}")
    print(f"Loss rate:      {loss_rate:.2%}")
    print(f"Draw rate:      {draw_rate:.2%}")
    print(f"Average reward: {average_reward:.3f}")

if __name__ == "__main__":

    train()
    average_reward = evaluate()
    print(f"Average reward: {average_reward}")