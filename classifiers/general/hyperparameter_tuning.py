from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetBinaryClassifier
import torch
from torch import nn, optim
from torch.utils.data import random_split

from data import EventDataset
from models import GeneralBinaryClassifier

db = EventDataset("../../data/background.csv", "../../data/signal.csv",
                  ["muon_inv_mass"], features_shape=(-1, 1), limit=20_000)
test_percent = 0.10
training_data, testing_data = random_split(db, [1 - test_percent, test_percent])

X = []
Y = []
for x, y in training_data:
	X.append([x])
	Y.append([y])

X = torch.Tensor(X)
Y = torch.Tensor(Y)

model = NeuralNetBinaryClassifier(
	module=GeneralBinaryClassifier,
	criterion=nn.BCEWithLogitsLoss,
	optimizer=optim.Adam,
	max_epochs=10,
	module__input_size=1,
)

param_grid = {
	"batch_size": [8, 16, 32, 64, 128],
	"module__hidden_layer_sizes": [
		[256, 256, 256],
		[256, 256, 128],
		[256, 128, 128],
		[512, 256, 128],
	],
	"optimizer__lr": [0.1, 0.01, 0.001, 0.0001]
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)

print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
