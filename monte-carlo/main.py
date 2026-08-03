import gymnasium as gym
from agent import MonteCarloAgent

env = gym.make("Blackjack-v1")
agent = MonteCarloAgent()

state, info = env.reset()

print("initial state: ", state)
terminated = False
truncated = False

episode = []


while not (terminated or truncated):

    action = agent.choose_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)

    episode.append([state, action, reward])
    state = next_state

returns = agent.calculate_returns(episode)


print("Episode: ")
for transition in episode:
    print(transition)

agent.update(episode)

print("\nLearned Q-values:")

for state_action, value in agent.Q.items():
    print(state_action, "=", value)






