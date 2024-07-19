import torch
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import BinaryAUROC

from matplotlib import pyplot as plt
import polars as pl

from classifiers.utils import train, test
from models import LorenzInvariantNetwork
from data.event_dataset import EventDataset

blur_size = 0.05
feature_cols = [
	"px_0", "py_0", "pz_0", "energy_0",
	"px_1", "py_1", "pz_1", "energy_1",
]
data = EventDataset("../../data/background.csv",
                    "../../data/signal.csv",
                    feature_cols,
                    features_shape=(-1, 2, 4),
                    limit=20_000,
                    blur_data=True,
                    blur_size=blur_size)

test_percent = 0.20
training_data, test_data = random_split(data, [1 - test_percent, test_percent])

batch_size = 128

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = LorenzInvariantNetwork(1, [512, 256, 128])

lr = 0.0001
weight_decay = 0.001
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)

epochs = 10
loss_over_time = []
accuracy_over_time = []
auc = []
max_acc = 0.0
max_acc_epoch = 0
metric = BinaryAUROC()
for t in range(epochs):
	print(f"Epoch {t + 1}\n-------------------------------")
	train(train_dataloader, model, loss_function, optimizer, True)
	loss, acc, auc_metric = test(test_dataloader, model, loss_function, True)
	
	loss_over_time.append(loss)
	accuracy_over_time.append(acc)
	auc.append(auc_metric)
	
	if acc > max_acc:
		torch.save(model, "model.pth")
		max_acc = acc
		max_acc_epoch = t + 1

print("Finished Training")
print(f"Model saved had {max_acc * 100:<0.2f}% accuracy, and was from epoch {max_acc_epoch}")

plt.plot(accuracy_over_time[0:max_acc_epoch])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Lorentz-invariant Classifier Accuracy per Epoch")
plt.show()

log = pl.DataFrame({
	"epoch": list(range(1, max_acc_epoch + 1)),
	"loss": loss_over_time[:max_acc_epoch],
	"acc": accuracy_over_time[:max_acc_epoch],
	"auc": auc,
	"model": "General PFN Classifier",
	"lr": lr,
	"weight_decay": weight_decay,
	"blur_size": blur_size
})
log.write_csv("log.csv")
print("Saved log.csv")
