import torch
from torch.utils.data import DataLoader, random_split

from classifiers.utils import train, test
from pfn_model import ParticleFlowNetwork
from data.event_dataset import EventDataset

data = EventDataset("../../data/background.csv", "../../data/signal.csv", limit=20_000)

test_percent = 0.10
training_data, test_data = random_split(data, [1 - test_percent, test_percent])

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = ParticleFlowNetwork(5,
                            10,
                            8,
                            [100, 100, 100],
                            [100, 100])

loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 10
for t in range(epochs):
	print(f"Epoch {t + 1}\n-------------------------------")
	train(train_dataloader, model, loss_function, optimizer)
	test(test_dataloader, model, loss_function)

print("Finished Training!\n")

do_save = input("Would you like to save the model? y/N -> ")
if do_save == "y":
	torch.save(model, "model.pth")
	print("Model Saved")
