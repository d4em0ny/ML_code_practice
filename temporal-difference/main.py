import gymnasium as gym
from agent import TDAgent

env = gym.make('Blackjack-v1')


agent = TDAgent()

no_of_episode = 100000

for no_of_ep in range(no_of_episode):
    state, _ = env.reset()
    truncated, terminated = False, False

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

for state_action, Value in agent.Q.items():
    print(f"state_action: {state_action}, \t Value: {Value}")