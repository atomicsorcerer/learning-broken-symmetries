import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torcheval.metrics import BinaryAUROC

import polars as pl

from classifiers.utils import train, test
from models import ParticleFlowNetwork
from data.event_dataset import EventDataset

results = []

BLUR_SIZE = 0.10
MAX_DB_SIZE = 100_000
TEST_PERCENT = 0.20
N_RUNS = 10
EPOCHS = 100

feature_cols = [
	"blurred_px_0", "blurred_py_0", "pz_0", "blurred_energy_0", "blurred_px_1", "blurred_py_1", "pz_1",
	"blurred_energy_1"
]
data = EventDataset("../../data/background.csv",
                    "../../data/signal.csv",
                    feature_cols,
                    features_shape=(-1, 2, 4),
                    limit=MAX_DB_SIZE,
                    blur_size=BLUR_SIZE,
                    shuffle_seed=314)

for i in range(N_RUNS):
	general_classifier_preference = None
	model = ParticleFlowNetwork(4,
	                            8,
	                            16,
	                            [512, 256, 128],
	                            [128, 128])
	
	lr = 0.00001
	weight_decay = 0.01
	loss_function = torch.nn.BCEWithLogitsLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=weight_decay)
	metric = BinaryAUROC()
	
	test_range = int(MAX_DB_SIZE * TEST_PERCENT)
	training_range = int(MAX_DB_SIZE * (1 - TEST_PERCENT) * ((i + 1) / N_RUNS))
	
	test_sampler = WeightedRandomSampler(data.norm_weights[0:test_range], test_range, replacement=True)
	training_sampler = WeightedRandomSampler(data.norm_weights[test_range:(test_range + training_range)],
	                                         training_range,
	                                         replacement=True)
	
	batch_size = 128
	
	test_dataloader = DataLoader(Subset(data, range(0, test_range)), batch_size=batch_size,
	                             sampler=test_sampler)
	train_dataloader = DataLoader(Subset(data, range(test_range, (test_range + training_range))), batch_size=batch_size,
	                              sampler=training_sampler)
	
	max_auc = 0
	for t in range(EPOCHS):
		print(f"Dataset Size = {training_range}\tEpoch {t + 1}\n-------------------------------")
		train(train_dataloader, model, loss_function, optimizer, True)
		loss, acc, auc_metric = test(test_dataloader, model, loss_function, metric, True)
		
		if auc_metric > max_auc:
			max_auc = auc_metric
	
	results.append({
		"dataset_size": training_range,
		"dataset_proportion": (i + 1) / N_RUNS,
		"final_auc": max_auc
	})

log = pl.DataFrame(results)
stamp = round(time.time())
log.write_csv(f"bulk training logs/auc_vs_db_size_{stamp}.csv")
print(f"Saved auc_vs_db_size_{stamp}.csv")
