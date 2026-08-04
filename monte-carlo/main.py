import gymnasium as gym
from agent import MCAgent

#create env from the one of Model env't found in Gymnasium
env = gym.make("Blackjack-v1")
# creating an agent.
agent = MCAgent()

#creating initial state. (reset the env and start as new)

no_of_episodes = 10000

for episode_no in range(no_of_episodes):
    state, info = env.reset()

    terminated = False
    truncated = False

    # creating a list to collect an episode
    episode = []

    #Generating Episodes
    while not (terminated):

        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        episode.append((state, action, reward))
        state = next_state

    agent.learn(episode)

print(len(agent.Q))

for transition, value in agent.Q.items():
    print(f"Transition: {transition}, Value: {value}")






