import torch
from torch.utils.data import DataLoader, random_split
from torcheval.metrics import BinaryAUROC

from matplotlib import pyplot as plt
import polars as pl

from classifiers.utils import train, test
from models import *
from data.event_dataset import EventDataset

blur_size = 0.10
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
                    blur_size=blur_size,
                    shuffle_seed=314)

test_percent = 0.20
training_data, test_data = random_split(data, [1 - test_percent, test_percent], torch.Generator().manual_seed(314))

batch_size = 128

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = LatentSpacePooledHybridClassifier(16,
                                          [128, 128, 128],
                                          [128, 128, 128],
                                          [64, 64, 64, 64, 64],
                                          [512, 256, 256, 128])

lr = 0.00001
weight_decay = 0.01
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)

epochs = 100
loss_over_time = []
accuracy_over_time = []
auc = []
max_acc = 0.0
max_acc_epoch = 0
metric = BinaryAUROC()
for t in range(epochs):
	print(f"Epoch {t + 1}\n-------------------------------")
	train(train_dataloader, model, loss_function, optimizer, True)
	loss, acc, auc_metric = test(test_dataloader, model, loss_function, metric, True)
	
	loss_over_time.append(loss)
	accuracy_over_time.append(acc)
	auc.append(auc_metric)
	
	if acc > max_acc:
		max_acc = acc
		max_acc_epoch = t + 1

print("Finished Training")

torch.save(model, "model.pth")
print("Saved Model")

print(f"Model saved had {max_acc * 100:<0.2f}% accuracy, and was from epoch {max_acc_epoch}.")

plt.plot(accuracy_over_time[0:max_acc_epoch])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Learned Weight Pooling Hybrid Classifier - Accuracy per Epoch")
plt.show()

log = pl.DataFrame({
	"epoch": list(range(1, epochs + 1)),
	"loss": loss_over_time,
	"acc": accuracy_over_time,
	"auc": auc,
	"model": "Latent Space Pooled Hybrid Classifier",
	"lr": lr,
	"weight_decay": weight_decay,
	"blur_size": blur_size
})
log.write_csv("log.csv")
print("Saved log.csv")
