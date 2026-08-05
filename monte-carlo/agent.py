import random as rnd

class MCAgent:
    gamma = 1
    epsilon = 0.1

    def __init__(self):
        self.Q = {}
        self.returns_sum = {}
        self.returns_count = {}

    def calculate_returns(self, episode):
        G = 0
        returns = []

        for _, _, R in reversed(episode):
            G = R + self.gamma * G

            returns.append(G)

        returns.reverse()
        return returns

    def choose_action(self, state):
        if rnd.random() < self.epsilon:
            return rnd.choice([0, 1])

        q_values = [
            self.Q.get((state, 0), 0),
            self.Q.get((state, 1), 0),
        ]

        max_val = max(q_values)

        best_action = [action for action, value in enumerate(q_values) if value == max_val]

        return rnd.choice(best_action)

    def learn(self, episode, returns):
        visited = set()

        for (state, action, reward), G in zip(episode, returns):
            if (state, action) in visited:
                continue

            visited.add((state, action))

            self.returns_sum[(state, action)] = self.returns_sum.get((state, action), 0) + G
            self.returns_count[(state, action)] = self.returns_count.get((state, action), 0) + 1

            self.Q[(state, action)] = self.returns_sum[(state, action)] / self.returns_count[(state, action)]

