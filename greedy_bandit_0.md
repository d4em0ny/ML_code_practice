
## What This code Demonstrates

* how a bandit environment works,
* how rewards are generated,
* how greedy action selection works,
* how estimated values are updated,
* why a greedy agent can get stuck on a bad arm,
* why exploration is needed.

---

## Core Idea

The agent keeps two important arrays:

* `estimate_values` → the agent’s current guess for each arm
* `counts` → how many times each arm has been selected

At each step:

1. The agent chooses the arm with the highest estimated value.
2. The environment returns a reward.
3. The agent updates its estimate for that arm.

---

## Why the Results Change Every Time

This project uses a **stochastic environment**.

That means the reward is random, even if the true probability stays fixed.

For example, if an arm has a true probability of `0.75`, it does **not** mean every pull returns reward `1`. It means that over many pulls, about 75% of the rewards should be `1`.

Because of this randomness:

* results may differ across runs,
* early rewards can mislead the greedy agent,
* estimates can look unstable when the number of steps is small.

This is a normal RL behavior and not a bug.

---

## Main Problem Faced

A pure greedy agent can suffer from **greedy lock-in**.

This happens when:

* all initial estimates are equal,
* the agent picks the first best arm,
* early random rewards make that arm look better than it really is,
* the agent keeps choosing the same arm and never explores others.

This is one of the main reasons greedy-only strategies are weak in reinforcement learning.

---

## Termination of the Problem

The issue is commonly known as:

* **Exploration-Exploitation Tradeoff**
* **Sampling Variance**
* **Greedy Lock-in**
* **Premature Convergence**

---

## Solution

The usual fix is **ε-greedy**.

With ε-greedy:

* most of the time the agent chooses the best-known arm,
* sometimes it chooses a random arm to explore.

This allows the agent to discover better arms instead of getting stuck too early.
