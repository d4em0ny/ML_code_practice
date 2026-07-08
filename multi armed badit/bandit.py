import random


true_probabilities = [
    0.10,
    0.35,
    0.55,
    0.20,
    0.75
]


def pull(arm):

    if random.random() < true_probabilities[arm]:
        return 1

    return 0



def greedy_action(estimate_values):

    return estimate_values.index(
        max(estimate_values)
    )



def update_estimate(
        estimate_values,
        counts,
        arm,
        reward):

    counts[arm] += 1

    estimate_values[arm] += (
        (1 / counts[arm])
        *
        (reward - estimate_values[arm])
    )



estimate_values = [0,0,0,0,0]

counts = [0,0,0,0,0]


for step in range(20):

    arm = greedy_action(estimate_values)

    reward = pull(arm)

    update_estimate(
        estimate_values,
        counts,
        arm,
        reward
    )


    print(
        "step:",
        step,
        "arm:",
        arm,
        "reward:",
        reward,
        "estimates:",
        estimate_values
    )