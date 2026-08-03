import random


class MonteCarloAgent:
    gamma = 1.0

    def __init__(self):
        self.Q = {}
        self.returns = {}
        self.returns_sum = {}
        self.returns_count = {}

    def choose_action(self, state):
        return random.choice([0, 1])

    def calculate_returns(self, episode):
        returns = []
        G = 0

        for _, _, reward in reversed(episode):
            G = reward + self.gamma * G
            returns.append(G)

        returns.reverse()
        return returns

    def update(self, episode):
        G =  0

        visited = set()

        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G
            state_action = (state, action)

            if state_action in visited:
                continue

            visited.add(state_action)


            self.returns_sum[state_action] = (
                    self.returns_sum.get(state_action, 0.0) + G
            )

            self.returns_count[state_action] = (
                    self.returns_count.get(state_action, 0) + 1
            )

            self.Q[state_action] = (
                    self.returns_sum[state_action] / self.returns_count[state_action]
            )
