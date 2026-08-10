import torch
import torch.nn as nn
from network import Network

online_network = Network()
target_network = Network()

target_network.load_state_dict(online_network.state_dict())

loss_fn = nn.MSELoss()

state = torch.tensor([1, 2, 3], dtype=torch.float32)
next_state = torch.tensor([0.89, 1.2, 2], dtype=torch.float32)

gamma = 1
action = 0
reward = 1

optimizer = torch.optim.Adam(online_network.parameters(), lr=0.01)

for i in range(100):
    q_values = online_network(state)
    q_value = q_values[action]

    with torch.no_grad():
        next_q_value = target_network(next_state)
        max_next_q_value = next_q_value.max()
    
        target = reward + gamma * max_next_q_value

    loss = loss_fn(q_value, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        target_network.load_state_dict(online_network.state_dict())


    print(f"step {i}: \tloss: {loss.item():2f} \tq_value: {q_value:2f} \ttarget: {target:2f}")
