import random

class MCAgent:
    epsilon = 0.1
    gamma = 1

    def __init__(self):
        self.Q = {}
        self.returns_sum = {}
        self.returns_count = {}

    def choose_action(self, state):
        #Explore
        if random.random() < self.epsilon:
            return random.choice([0, 1])

        q_values = [
            self.Q.get((state, 0), 0),
            self.Q.get((state, 1), 0),
        ]

        max_q = max(q_values)

        best_actions = [
            action
            for action, q in enumerate(q_values)
            if q == max_q
        ]

        return random.choice(best_actions)

    """
        for each (state, action, reward). 
        calculate their reward starting from the back
    """
    def calculate_returns(self, episode):
        returns = []
        G = 0

        for _, _, reward in reversed(episode):
            G = reward + self.gamma * G
            returns.append(G)

        returns.reverse()
        return returns

    def learn(self, episode):
        returns = self.calculate_returns(episode)

        visited = set()

        for (state, action, reward), G in zip(episode, returns):

            state_action = (state, action)

            if state_action in visited:
                continue

            visited.add(state_action)

            self.returns_sum[state_action] = self.returns_sum.get(state_action, 0) + G
            self.returns_count[state_action] = self.returns_count.get(state_action, 0) + 1

            # Q(St, At) = Gt / |Gt|
            self.Q[state_action] = self.returns_sum[state_action] / self.returns_count[state_action]



