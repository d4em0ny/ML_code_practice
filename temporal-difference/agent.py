import random as rnd

class TDAgent:
    alpha = 0.1
    gamma = 1
    epsilon = 0.1

    def __init__(self):
        self.Q = {}

    def get_value(self, state, action):
        return self.Q.get((state, action), 0)

    def learn(self, state, action, reward, next_state, next_action, terminated, truncated):
        current_q = self.Q.get((state, action), 0.0)

        if terminated or truncated:
            target = reward

        else:
            next_q = self.Q.get((next_state, next_action), 0.0)
            target = reward + self.gamma * next_q


        self.Q[(state, action)] = current_q + self.alpha * (target - current_q)

    def choose_action(self, state):
        if rnd.random() < self.epsilon:
            return rnd.choice([0, 1])

        q_values = [
            self.Q.get((state, 0), 0.0),
            self.Q.get((state, 1), 0.0)
        ]

        max_q = max(q_values)

        best_action = [
            action for action, q in enumerate(q_values) if q == max_q
        ]

        return rnd.choice(best_action)