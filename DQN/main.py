import gymnasium as gym
import torch
import torch.nn as nn
import random as rand

from network import Network
from replay_buffer import ReplayBuffer


# Networks
target_network = Network()
online_network = Network()

target_network.load_state_dict(online_network.state_dict())


# Environment
env = gym.make("Blackjack-v1")


# Hyperparameters
gamma = 1.0
epsilon = 0.1

batch_size = 32
buffer_capacity = 10_000

target_update_frequency = 100


replay_buffer = ReplayBuffer(buffer_capacity)


optimizer = torch.optim.Adam(online_network.parameters(),lr=0.001)
loss_function = nn.MSELoss()



def encode_state(state):

    player_sum, dealer_card, usable_ace = state

    return torch.tensor(
        [
            player_sum / 21.0,
            dealer_card / 10.0,
            float(usable_ace)
        ],
        dtype=torch.float32
    )


# Action Selection
def choose_action(state):

    if rand.random() < epsilon:
        return rand.choice([0, 1])

    state_tensor = encode_state(state)

    with torch.no_grad():
        q_values = online_network(state_tensor)

    return q_values.argmax().item()



def train():

    num_episodes = 10000
    episode_rewards = []

    for episode in range(num_episodes):

        state, _ = env.reset()

        terminated = False
        truncated = False

        total_reward = 0

        while not (terminated or truncated):

            # 1. Choose action
            action = choose_action(state)


            # 2. Interact with environment

            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated


            # 3. Store experience

            replay_buffer.push((state, action, reward, next_state, done))


            # 4. Train only if enough
            #    experiences exist

            if len(replay_buffer) >= batch_size:

                # Get random batch
                batch = replay_buffer.sample(batch_size)

                # 5. Separate the batch
                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []

                for experience in batch:

                    state_b, action_b, reward_b, next_state_b, done_b = experience

                    states.append(encode_state(state_b))
                    actions.append(action_b)
                    rewards.append(reward_b)
                    next_states.append(encode_state(next_state_b))
                    dones.append(done_b)

                # 6. Convert to tensors

                states = torch.stack(states)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.stack(next_states)
                dones = torch.tensor(dones, dtype=torch.bool)

                # 7. Current Q-values

                q_values = online_network(states)
                q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)


                # 8. Target Q-values
                with torch.no_grad():

                    next_q_values = target_network(next_states)
                    max_next_q_values = next_q_values.max(dim=1).values

                    targets = rewards + (gamma * max_next_q_values * (~dones))

                # 9. Calculate loss
                loss = loss_function(q_values, targets)


                # 10. Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state


        # End of episode
        episode_rewards.append(total_reward)


        # Update target network
        if (episode + 1) % target_update_frequency == 0:

            target_network.load_state_dict(online_network.state_dict())



        # Print progress

        if (episode + 1) % 1000 == 0:

            avg_reward = (sum(episode_rewards[-1000:]) / 1000)

            print(
                f"Episode: {episode + 1}, "
                f"Average reward: {avg_reward:.3f}"
            )


# Evaluation

def evaluate(num_episodes=10_000):

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

            action = q_values.argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)

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

    print("\nDQN")

    evaluate()