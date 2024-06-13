import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split

from classifiers.utils import train, test
from models import GeneralBinaryClassifier
from data.event_dataset import EventDataset

data = EventDataset("../../data/background.csv", "../../data/signal.csv",
                    ["muon_inv_mass"], features_shape=(-1, 1), limit=20_000)

test_percent = 0.10
training_data, test_data = random_split(data, [1 - test_percent, test_percent])

batch_size = 16

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = GeneralBinaryClassifier(1, [128, 128], True)

loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-8)

epochs = 50
loss_over_time = []
for t in range(epochs):
	print(f"Epoch {t + 1}\n-------------------------------")
	train(train_dataloader, model, loss_function, optimizer)
	loss, acc = test(test_dataloader, model, loss_function)
	
	loss_over_time.append(loss)

plt.plot(loss_over_time)
plt.show()

print("Finished Training!\n")

do_save = input("Would you like to save the model? y/N -> ")
if do_save == "y":
	torch.save(model, "model.pth")
	print("Model Saved")
