import gymnasium as gym

from agent import MCAgent

env = gym.make("Blackjack-v1")
agent = MCAgent()


num_of_episodes = 10000

for no_of_ep in range(num_of_episodes):

    state, info = env.reset()

    terminated, truncated = False, False
    episode = []

    while not (terminated or truncated):
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        episode.append((state, action, reward))
        state = next_state

    returns = agent.calculate_returns(episode)
    agent.learn(episode, returns)


for state_action, Value in agent.Q.items():
    print(f"state_action pair: {state_action}, \tValue: {Value}")



