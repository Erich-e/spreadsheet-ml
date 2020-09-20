import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(4, 3)
		self.fc2 = nn.Linear(3, 3)
		self.fc3 = nn.Linear(3, 3)
	
	def forward(self, x):
		x = torch.tanh(self.fc1(x))
		x = torch.tanh(self.fc2(x))
		x = torch.tanh(self.fc3(x))
		return x

net = Net()

# get data
train = pd.read_csv("iris_train.csv")
test = pd.read_csv("iris_test.csv")

def split(data):
	inputs = torch.tensor(data[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values, dtype=torch.float)
	labels = torch.tensor(data[["setosa", "versicolor", "virginica"]].values, dtype=torch.float)
	return inputs, labels

# Train
BATCH_SIZE = 10
NUM_BATCHES = len(train) / BATCH_SIZE

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

start_time = time.time()

for epoch in range(10):
	running_loss = 0
	for i, data in enumerate(np.array_split(train, NUM_BATCHES)):
		inputs, labels = split(data)

		optimizer.zero_grad()

		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

	print(f"Epoch {epoch+1:<3} loss: {running_loss / NUM_BATCHES:4f} time taken: {time.time() - start_time:4f}")


# Test
inputs, labels = split(test)
outputs = net(inputs)
_, predictions = torch.max(outputs.data, 1)
_, actual = torch.max(labels.data, 1)

num_correct = (predictions == actual).sum().item()
accuracy = num_correct / len(labels)

print(f"Accuracy: {accuracy:.4f}")
