import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torcheval.metrics import BinaryAUROC

import polars as pl

from classifiers.utils import train, test
from models import LatentSpacePooledHybridClassifier
from data.event_dataset import EventDataset

results = []

BLUR_SIZE = 0.10
MAX_DB_SIZE = 100_000
TEST_PERCENT = 0.20
N_RUNS = 10
EPOCHS = 100

for i in range(N_RUNS):
	general_classifier_preference = None
	model = LatentSpacePooledHybridClassifier(16,
	                                          [128],
	                                          [128],
	                                          [64, 64],
	                                          [512, 256, 128],
	                                          general_classifier_preference=general_classifier_preference)
	
	lr = 0.00001
	weight_decay = 0.01
	loss_function = torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
	metric = BinaryAUROC()
	
	feature_cols = [
		"blurred_px_0", "blurred_py_0", "pz_0", "blurred_energy_0", "blurred_px_1", "blurred_py_1", "pz_1",
		"blurred_energy_1"
	]
	db_limit = int(((i + 1) / N_RUNS) * MAX_DB_SIZE)
	data = EventDataset("../../data/background.csv",
	                    "../../data/signal.csv",
	                    feature_cols,
	                    features_shape=(-1, 2, 4),
	                    limit=db_limit,
	                    blur_size=BLUR_SIZE,
	                    shuffle_seed=314)
	
	test_percent = 0.20
	training_range = int(len(data) * (1 - test_percent))
	test_range = int(len(data) * test_percent)
	
	training_sampler = WeightedRandomSampler(data.norm_weights[0:training_range], training_range, replacement=True)
	test_sampler = WeightedRandomSampler(data.norm_weights[training_range:], test_range, replacement=True)
	
	batch_size = 128
	
	train_dataloader = DataLoader(Subset(data, range(0, training_range)), batch_size=batch_size,
	                              sampler=training_sampler)
	test_dataloader = DataLoader(Subset(data, range(training_range, len(data))), batch_size=batch_size,
	                             sampler=test_sampler)
	
	for t in range(EPOCHS):
		print(f"Dataset Size = {db_limit}\tEpoch {t + 1}\n-------------------------------")
		train(train_dataloader, model, loss_function, optimizer, True)
		loss, acc, auc_metric = test(test_dataloader, model, loss_function, metric, True)
		
		if t == EPOCHS - 1:
			results.append({
				"dataset_size": db_limit,
				"dataset_proportion": (i + 1) / N_RUNS,
				"final_auc": auc_metric,
				"final_acc": acc,
				"final_loss": loss
			})

log = pl.DataFrame(results)
log.write_csv(f"auc_vs_db_size_{stamp}.csv")
print(f"Saved auc_vs_db_size_{stamp}.csv")
