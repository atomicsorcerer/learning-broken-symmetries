from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetBinaryClassifier
import torch
from torch import nn, optim
from torch.utils.data import random_split

from data import EventDataset
from models import WeightedHybridClassifier

feature_cols = [
	"px_0", "py_0", "pz_0", "energy_0",
	"px_1", "py_1", "pz_1", "energy_1",
]
db = EventDataset("../../data/background.csv",
                  "../../data/signal.csv",
                  feature_cols,
                  features_shape=(-1, 2, 4),
                  limit=20_000)

test_percent = 0.01
training_data, testing_data = random_split(db, [1 - test_percent, test_percent])

X = []
Y = []
for x, y in training_data:
	X.append(x.numpy())
	Y.append(y)

X = torch.Tensor(X)
Y = torch.Tensor(Y)

model = NeuralNetBinaryClassifier(
	module=WeightedHybridClassifier,
	criterion=nn.BCEWithLogitsLoss,
	optimizer=optim.AdamW,
	max_epochs=10,
	module__pfn_mapping_hidden_layer_dimensions=[128, 128],
	module__pfn_classifier_hidden_layer_dimensions=[512, 256, 128],
	module__lorenz_invariant_hidden_layer_dimensions=[512, 256, 128],
	module__latent_space_dim=16
)

param_grid = {
	"batch_size": [8, 16, 32, 64, 128],
	"optimizer__lr": [10 ** (-i) for i in range(8)],
	"module__invariant_network_weight": [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)

print(grid_result)
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
