from jinja2 import optimizer

from network import Network
import torch
import torch.nn as nn

network = Network()

state = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([0, 1], dtype=torch.float32)

loss_fn = nn.MSELoss()

for i in range(1000):
    prediction = network(state)

    loss = loss_fn(prediction, target)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if i % 10 == 0:
        print(f"loss {loss.item():4f} \t prediction: {prediction} \ttarget: {target}")

