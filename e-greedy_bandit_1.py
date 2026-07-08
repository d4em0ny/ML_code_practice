import random

'''
    should not be defined necesserly in real world env't
    we are using it just to evaluate how the agent leanring process is
'''
true_probabilities = [
    0.10,
    0.35,
    0.55,
    0.20,
    0.75
]

'''
    FUNCTION TO GIVE REWARD
    Assume the random.random() is equal to 0.3 ...
    our Highest probability from the true_probability is 75% ...
    so only for the probability of 0 to 0.75 it will return 1
    otherwise for the rest 25% which is 0.75 to 1. it will return 0

'''
def pull(arm):
    if random.random() < true_probabilities[arm]:
        return 1

    return 0


epsilon = 0.1
def epsilon_greedy_action(estimate_values):
    rand = random.random()

    if rand < epsilon:
        return random.randint(0, 4)
    
    return estimate_values.index(
        max(estimate_values)
    )





def update_estimate(
        estimate_values,
        counts,
        arm,
        reward):

    counts[arm] += 1
    estimate_values[arm] += ( (1 / counts[arm]) * (reward - estimate_values[arm]))


estimate_values = [0,0,0,0,0]
counts = [0,0,0,0,0]


for step in range(10000):

    arm = epsilon_greedy_action(estimate_values)
    reward = pull(arm)

    update_estimate(
        estimate_values,
        counts,
        arm,
        reward
    )


    print(
        "step:", step,
        "arm:", arm,
        "reward:", reward,
        "estimates:", estimate_values
    )