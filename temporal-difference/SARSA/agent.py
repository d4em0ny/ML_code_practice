import random as rnd

class QLearningAgent():
    gamma = 1
    epsilon = 0.1
    alpha = 0.5

    def __init__(self):
        self.Q = {}

    def get_qvalue(self, state, action):
        return self.Q.get((state, action), 0)

    def choose_action(self, state):
        if rnd.random() < self.epsilon:
            return rnd.choice([0, 1])

        q_values = [
            self.Q.get((state, 0), 0),
            self.Q.get((state, 0), 0),
        ]

        max_value = max(q_values)

        best_action = [ action for action, value in enumerate(q_values) if value == max_value ]

        return rnd.choice(best_action)

    def learn(self, state, action, reward, next_state, next_action, terminated, truncated    ):
        current_q = self.get_qvalue(state, action)

        if terminated or truncated:
            target = reward
        else:
            next_q = self.get_qvalue(next_state, next_action)
            target = reward + self.gamma * next_q

        self.Q[(state, action)] = current_q + (self.alpha * (target - current_q))




