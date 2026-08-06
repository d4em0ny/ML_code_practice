import gymnasium as gym
from agent import QLearningAgent


env = gym.make("Blackjack-v1")
agent = QLearningAgent()


state, _ = env.reset()

terminated = False
truncated = False

action = agent.choose_action(state)

while not (terminated or truncated):

    next_state, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        next_action = None
    else:
        next_action = agent.choose_action(next_state)

    agent.learn(state, action, reward, next_state, next_action, terminated, truncated)

    state = next_state

    if next_action is not None:
        action = next_action


print(agent.Q)