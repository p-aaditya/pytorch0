import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from torchmetrics import Accuracy
import torchvision
from torchvision.transforms import ToTensor
from timeit import default_timer as timer
from tqdm.auto import tqdm
from torch import nn
import zipfile
import os
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pathlib
from typing import Tuple, Dict, List
from torchinfo import summary


## Version ##
print(torch.__version__)
#############

## Tensors ##
## Tensors ##
## Tensors ##

# Scalars #
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)
print(scalar.item())
###########

# Vectors #
vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim)
print(vector.shape)
###########

# Matrix #
matrix = torch.tensor([[7, 8],
					   [9, 10]])
print(matrix)
print(matrix.ndim)
print(matrix.shape)
##########

# tensor #
tensor = torch.tensor([[1, 2, 3],
					   [4, 5, 6],
					   [7, 8, 9]])
print(tensor)
print(tensor.ndim)
print(tensor.shape)
##########

# Random tensors #
random_tensor = torch.rand(size = (3, 4))
print(random_tensor, random_tensor.dtype)
print(random_tensor.ndim, random_tensor.shape)
##################

# Zeros and ones #
zeros = torch.zeros(size = (3, 4))
ones = torch.ones(size = (3, 4))
print(zeros, zeros.dtype)
print(ones, ones.dtype)
##################

# tensor range #
zero_to_ten = torch.arange(0, 10)
zero_to_ten = torch.arange(start = 0, end = 10, step = 1)
print(zero_to_ten)
ten_zeros = torch.zeros_like(input = zero_to_ten)
print(ten_zeros)
################

# Tensor datatypes #
print("https://pytorch.org/docs/stable/tensors.html#data-types")
####################

# Manipulating Tensors #
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)
print(tensor * 10)
print(tensor - 10)
print(tensor / 10)
print(torch.mul(tensor ,10))
########################

# Matrix Multiplication #
print(tensor * tensor)
print(torch.matmul(tensor, tensor))
print(tensor @ tensor) # not recommended
#########################

# Torch operations #
x = torch.arange(0, 100, 10)
print(x)
print(x.min())
print(x.max())
print(x.type(torch.float32).mean())
print(x.sum())
print(x.argmax())
####################

# Change torch datatype #
tensor = torch.arange(10., 100., 10.)
print(torch.dtype)
tensor_float16 = tensor.type(torch.float16)
tensor_int8 = tensor.type(torch.int8)
print(tensor_float16)
print(tensor_int8)
#########################

# Reshape Stacking Squeezing Unsqueezing #
x = torch.arange(1., 8.)
print(x, x.shape)
x_reshaped = x.reshape(1, 7)
print(x_reshaped, x_reshaped.shape)
z = x.view(1, 7)
print(z, z.shape)
x_stacked = torch.stack([x, x, x, x], dim = 0)
print(x_stacked)
x_squeezed = x_reshaped.squeeze()
print(x_squeezed, x_squeezed.shape)
x_unsqueezed = x_squeezed.unsqueeze(dim = 0)
print(x_unsqueezed)
x_original = torch.rand(size = (224, 224, 3))
x_permuted = x_original.permute(2, 0, 1)
print(x_original.shape, x_permuted.shape)
##########################################

# Indexing #
x = torch.arange(1, 10).reshape(1, 3, 3)
print(x, x.shape)
print(x[0])
print(x[0][0])
print(x[0][0][0])
print(x[:, 0])
print(x[:, :, 1])
############

# Numpy #
array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(array, tensor)
tensor = torch.ones(7)
print(tensor.numpy())
#########

# Reproduciblity #
random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)
print(random_tensor_A == random_tensor_B)
RANDOM_SEED=42
torch.manual_seed(seed=RANDOM_SEED) 
random_tensor_C = torch.rand(3, 4)
torch.random.manual_seed(seed=RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)
print(random_tensor_C == random_tensor_D)
##################

# GPU #
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(torch.cuda.device_count())
tensor = torch.tensor([1, 2, 3])
print(tensor, tensor.device)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu)
#######

#############
#############
#############


# Workflow #
# Workflow #
# Workflow #

# Initialization #
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.02
x = torch.arange(start, end, step).unsqueeze(dim = 1)
y = weight * x + bias
print(x[:10], y[:10])
train_split = int(0.8 * len(x))
x_train, y_train, x_test, y_test = x[:train_split], y[:train_split], x[train_split:], y[train_split:]
print(len(x_train), len(y_train), len(x_test), len(y_test))
##################

# Visualization #
def plot_predictions(train_data=x_train, 
                     train_labels=y_train, 
                     test_data=x_test, 
                     test_labels=y_test, 
                     predictions=None):

	plt.figure(figsize=(10, 7))
	plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
	plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
	if predictions is not None:
		plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
	plt.legend(prop={"size": 14})
	# plt.show()

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
#plot_predictions()
#################

# Class definition #
class LinearRegressionModel(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        self.weights = torch.nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)
        self.bias = torch.nn.Parameter(torch.randn(1, dtype=torch.float),requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

####################
# every model must be a subclass of nn.Module #
# initialize weight and bias #
# nn.Parameter accepts tensors that are stored by nn.Module #
# every model must have a forward method which does the computation #
# will accept and return tensor type only #
####################

# time pass but required #
# or predicting as well #
torch.manual_seed(42)
model_0 = LinearRegressionModel()
print(list(model_0.parameters()))
print(model_0.state_dict())

with torch.inference_mode():
	y_preds = model_0(x_test)

print(f"Number of testing samples: {len(x_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

# plot_predictions(predictions=y_preds)
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.01)
##########################

# training #
torch.manual_seed(42)
epochs = 100
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()

    with torch.inference_mode():
      test_pred = model_0(x_test)
      test_loss = loss_fn(test_pred, y_test.type(torch.float))
      if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

# plt.plot(epoch_count, train_loss_values, label="Train loss")
# plt.plot(epoch_count, test_loss_values, label="Test loss")
# plt.title("Training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()

print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")
############

# Saving and loading #
model_path = Path("models")
model_path.mkdir(parents = True, exist_ok = True)
model_name = "pytorch_workflow.pth"
model_save_path = model_path / model_name
print(f"/saving model to: {model_save_path}")
torch.save(obj = model_0.state_dict(), f = model_save_path)

loaded_model_0 = LinearRegressionModel()
loaded_model_0.load_state_dict(torch.load(f = model_save_path))

loaded_model_0.eval()
with torch.inference_mode():
	loaded_model_pred = loaded_model_0(x_test)
print(test_pred == loaded_model_pred)
######################

# cuda #
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
########

# repeatu #
weight = 0.7
bias = 0.3
start = 0
end = 1
step = 0.02

x = torch.arange(start, end, step).unsqueeze(dim  = 1)
y = weight * x + bias
print(x[:10], y[:10])

train_split = int(0.8 * len(x))
x_train, x_test, y_train, y_test = x[:train_split], x[train_split:], y[:train_split], y[train_split:]
print(len(x_train), len(x_test), len(y_train), len(y_test))
# plot_predictions()

class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_features = 1, out_features = 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
model_1 = LinearRegressionModel()
print(model_1, model_1.state_dict())
print(next(model_1.parameters()).device)

# GPU #
# GPU #
model_1.to(device)
#######
#######

loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

torch.manual_seed(42)
epochs = 1000 
x_train = x_train.to(device)
x_test = x_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    model_1.train()
    y_pred = model_1(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_1.eval()

    with torch.inference_mode():
        test_pred = model_1(x_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

model_1.eval()
with torch.inference_mode():
    y_preds = model_1(x_test)
# plot_predictions(predictions=y_preds.cpu())

# Load and save model #
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "01_pytorch_workflow_model_1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH) 

loaded_model_1 = LinearRegressionModel()
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))
loaded_model_1.to(device)
print(f"Loaded model:\n{loaded_model_1}")
print(f"Model on device:\n{next(loaded_model_1.parameters()).device}")
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(x_test)
print(y_preds == loaded_model_1_preds)
#######################
###########

# Non-linear Datasets #
n_samples = 1000
x, y = make_circles(n_samples, noise = 0.03, random_state = 42)
circles = pd.DataFrame({"x1" : x[:, 0],
                        "x2" : x[:, 1],
                        "label" : y})
# plt.scatter(x = x[:, 0],
#             y = x[:, 1],
#             c = y,
#             cmap = plt.cm.RdYlBu)
x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

# Model #
class CircleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(in_features = 2, out_features = 5)
        self.layer_2 = torch.nn.Linear(in_features = 5, out_features = 1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))
#########

# Very Good Practise #
# Very Good Practise #
model_0 = CircleModel().to(device)
untrained_preds = model_0(x_test.to(device))
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.1)

y_logits = model_0(x_test.to(device))[:5]
print(y_logits)
y_pred_probs = torch.sigmoid(y_logits)
print(y_pred_probs)
y_preds = torch.round(y_pred_probs)
y_pred_labels = torch.round(torch.sigmoid(model_0(x_test.to(device))[:5]))
print(torch.eq(y_preds.squeeze(), y_pred_labels.squeeze()))
print(y_preds.squeeze())
######################
######################

# repeatu #
torch.manual_seed(42)
epochs = 100
x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)

for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)
    if epoch % 10 == 0:
        print(f"Epochs : {epoch} | Loss : {loss:.5f}, Accuracy : {acc:.2f}% | Test loss : {test_loss:.5f}, Test acc : {test_acc:.2f}%")

# plt.figure(figsize = (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_0, x_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model_0, x_test, y_test)
# plt.show()
###########

# Model #
class Circle_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(in_features = 2, out_features = 10)
        self.layer_2 = torch.nn.Linear(in_features = 10, out_features = 10)
        self.layer_3 = torch.nn.Linear(in_features = 10, out_features = 1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))

model_1 = Circle_Model().to(device)
print(model_1)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_1.parameters(), lr = 0.1)
#########

# repeatu #
torch.manual_seed(42)
epochs = 1000
x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)

for epoch in range(epochs):
    model_1.train()
    y_logits = model_1(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)
    if epoch % 100 == 0:
        print(f"Epochs : {epoch} | Loss : {loss:.5f}, Accuracy : {acc:.2f}% | Test loss : {test_loss:.5f}, Test acc : {test_acc:.2f}%")

# plt.figure(figsize = (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_1, x_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model_1, x_test, y_test)
# plt.show()
###########

# Model #
class Circle_Model_(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(in_features = 2, out_features = 10)
        self.layer_2 = torch.nn.Linear(in_features = 10, out_features = 10)
        self.layer_3 = torch.nn.Linear(in_features = 10, out_features = 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.layer_1(x))))

model_2 = Circle_Model_().to(device)
print(model_2)
#########

# repeatu #
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_2.parameters(), lr = 0.1)

for epoch in range(epochs):
    model_2.train()
    y_logits = model_2(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_2.eval()
    with torch.inference_mode():
        test_logits = model_2(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)
    if epoch % 100 == 0:
        print(f"Epochs : {epoch} | Loss : {loss:.5f}, Accuracy : {acc:.2f}% | Test loss : {test_loss:.5f}, Test acc : {test_acc:.2f}%")

model_2.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_2(x_test))).squeeze()
print(y_preds[: 10], y[: 10])

# plt.figure(figsize = (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_2, x_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model_2, x_test, y_test)
# plt.show()
###########
#######################

# Multi-class Classification #
num_classes = 4 
num_features = 2 
random_seed = 42

# Dataset #
x_blob, y_blob = make_blobs(n_samples = 1000, n_features = num_features, centers = num_classes,
                            cluster_std = 1.5, random_state = random_seed)
x_blob = torch.from_numpy(x_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
x_train, x_test, y_train, y_test = train_test_split(x_blob, y_blob, test_size = 0.2, random_state = 42)
# plt.figure(figsize = (10, 7))
# plt.scatter(x_blob[:, 0], x_blob[:, 1], c = y_blob, cmap = plt.cm.RdYlBu)
# plt.show()
###########

# Model #
class Blob(torch.nn.Module):
    def __init__(self, input_features, output_features, hidden_units = 8):
        super().__init__()
        self.linear_layer_stack = torch.nn.Sequential(torch.nn.Linear(in_features = input_features, out_features = hidden_units),
                                                # torch.nn.ReLU(),
                                                torch.nn.Linear(in_features = hidden_units, out_features = hidden_units),
                                                # torch.nn.ReLU(),
                                                torch.nn.Linear(in_features = hidden_units, out_features = output_features))

    def forward(self, x):
        return self.linear_layer_stack(x)

model_3 = Blob(input_features = num_features, output_features = num_classes).to(device)
print(model_3)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_3.parameters(), lr = 0.1)
print(model_3(x_train.to(device))[: 5])
#########

# repeatu #
torch.manual_seed(42)
epochs = 100
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_3.train()
    y_logits = model_3(x_train)
    y_pred = torch.softmax(y_logits, dim = 1).argmax(dim = 1)
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true = y_train, y_pred = y_pred)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(x_test)
        test_pred = torch.softmax(test_logits, dim = 1).argmax(dim = 1)
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true = y_test, y_pred = test_pred)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

model_3.eval()
with torch.inference_mode():
    y_logits = model_3(x_test)
y_preds = torch.softmax(y_logits, dim = 1).argmax(dim = 1)
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_3, x_train, y_train)
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model_3, x_test, y_test)
# plt.show()
###########
##############################

# Accuracy #
acc = Accuracy(task = "multiclass", num_classes = 4).to(device)
print(acc(y_preds, y_test))
############

# Images #
# Dataset #
train = torchvision.datasets.FashionMNIST(root = "data", train = True, download = True, transform = ToTensor(), target_transform = None)
test = torchvision.datasets.FashionMNIST(root = "data", train = False, download = True, transform = ToTensor())
image, label = train[0]
print(image, label)
print(image.shape)
print(len(train.data), len(train.targets), len(test.data), len(test.targets))
class_names = train.classes
print(class_names)
###########

# Visualisation #
image, label = train[0]
print(f"Image shape: {image.shape}")
# plt.imshow(image.squeeze())
# plt.title(label)
# plt.show()
torch.manual_seed(42)
fig = plt.figure(figsize = (9, 9))
rows, cols = 4, 4

# for i in range(1, rows * cols + 1):
#     random_idx = torch.randint(0, len(train), size = [1]).item()
#     img, label = train[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap = "gray")
#     plt.title(class_names[label])
#     plt.axis(False)
# plt.show()
#################

# DataLoader #
batch_size = 32
train_dataloader = DataLoader(train, batch_size = batch_size, shuffle = True)
test_dataloader = DataLoader(test, batch_size = batch_size, shuffle = False)
print(f"DataLoaders: (train, test)")
print(f"Length of train: {len(train_dataloader)}")
print(f"Length of test: {len(test_dataloader)}")
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_labels_batch.shape)
##############

# Visualisation #
torch.manual_seed(47)
random_idx = torch.randint(0, len(train_features_batch), size = [1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
# plt.imshow(img.squeeze(), cmap = "gray")
# plt.title(class_names[label])
# plt.axis("Off")
print(f"Image Size: {img.shape}")
print(f"Label: {label}, label_size: {label.shape}")
# plt.show()
#################

# Model #
class Fashion_MNIST(torch.nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(torch.nn.Flatten(),
                                         torch.nn.Linear(in_features = input_shape, out_features = hidden_units),
                                         torch.nn.Linear(in_features = hidden_units, out_features = output_shape))

    def forward(self, x):
        return self.layer_stack(x)

#########

torch.manual_seed(47)
model_0 = Fashion_MNIST(input_shape = 784, hidden_units = 10, output_shape = len(class_names))
model_0.to("cpu")
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_0.parameters(), lr = 0.1)

def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

# different repeatu #
torch.manual_seed(47)
train_time_start_on_cpu = timer()
epochs = 3

for epoch in range(epochs):
    print(f"Epoch: {epoch}\n-----------")
    train_loss = 0 

    for batch, (x,y) in enumerate(train_dataloader):
        model_0.train()
        y_pred = model_0(x)
        loss = loss_fn(y_pred, y)
        train_loss += loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(x)}/{len(train_dataloader.dataset)} samples")
    train_loss /= len(train_dataloader)
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for x, y in test_dataloader:
            test_pred = model_0(x)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true = y, y_pred = test_pred.argmax(dim = 1))
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    print(f"Train Loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.3f}%")

train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(train_time_start_on_cpu, train_time_end_on_cpu, device = str(next(model_0.parameters()).device))
#####################

torch.manual_seed(47)

def eval_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for x, y in data_loader:
            y_pred = model(x)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true = y, y_pred = y_pred.argmax(dim = 1))
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, "model_loss": loss.item(), "model_acc": acc}

model_0_results = eval_model(model_0, test_dataloader, loss_fn, accuracy_fn)
print(model_0_results)

# Model #
class Fashion_MNIST_1(torch.nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(torch.nn.Flatten(),
                                               torch.nn.Linear(in_features = input_shape, out_features = hidden_units),
                                               torch.nn.ReLU(),
                                               torch.nn.Linear(in_features = hidden_units, out_features = output_shape),
                                               torch.nn.ReLU())

    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(47)
model_1 = Fashion_MNIST_1(784, 10, len(class_names)).to(device)
print(next(model_1.parameters()).device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_1.parameters(), lr = 0.1)
#########

# Train and test function #
def train_step(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,
               accuracy_fn, device: torch.device = device):
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        train_loss += loss 
        train_acc += accuracy_fn(y_true = y, y_pred = y_pred.argmax(dim = 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader, model: torch.nn.Module, loss_fn: torch.nn.Module, accuracy_fn, device: torch.device = device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() 
    with torch.inference_mode(): 
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            test_pred = model(x)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

###########################

torch.manual_seed(42)
train_time_start_on_gpu = timer()
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, model=model_1, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn)
    test_step(data_loader=test_dataloader, model=model_1, loss_fn=loss_fn, accuracy_fn=accuracy_fn)

train_time_end_on_gpu = timer()
total_train_time_model_1 = print_train_time(start=train_time_start_on_gpu, end=train_time_end_on_gpu, device=device)
torch.manual_seed(42)

# Evaluate the model # 
def eval_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn, device: torch.device = device):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__,  "model_loss": loss.item(), "model_acc": acc}

model_1_results = eval_model(model=model_1, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn, device=device)
print(model_1_results) 
######################

# CNN Model #
class FashionMNISTModelV2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1, hidden_units=10, output_shape=len(class_names)).to(device)
print(model_2)
#############

# Testing for one sample #
torch.manual_seed(42)
images = torch.randn(size=(32, 3, 64, 64))
test_image = images[0]
conv_layer = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0)
conv_layer(test_image) 
torch.manual_seed(42)
conv_layer_2 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(5, 5), stride=2, padding=0)
conv_layer_2(test_image.unsqueeze(dim=0)).shape
print(f"Image batch shape: {images.shape} -> [batch_size, color_channels, height, width]")
print(f"Single image shape: {test_image.shape} -> [color_channels, height, width]") 
print(f"Single image pixel values:\n{test_image}")
print(f"Test image original shape: {test_image.shape}")
print(f"Test image with unsqueezed dimension: {test_image.unsqueeze(dim=0).shape}")
max_pool_layer = nn.MaxPool2d(kernel_size=2)
test_image_through_conv = conv_layer(test_image.unsqueeze(dim=0))
print(f"Shape after going through conv_layer(): {test_image_through_conv.shape}")
test_image_through_conv_and_max_pool = max_pool_layer(test_image_through_conv)
print(f"Shape after going through conv_layer() and max_pool_layer(): {test_image_through_conv_and_max_pool.shape}")
torch.manual_seed(42)
random_tensor = torch.randn(size=(1, 1, 2, 2))
print(f"Random tensor:\n{random_tensor}")
print(f"Random tensor shape: {random_tensor.shape}")
max_pool_layer = nn.MaxPool2d(kernel_size=2)
max_pool_tensor = max_pool_layer(random_tensor)
print(f"\nMax pool tensor:\n{max_pool_tensor} <- this is the maximum value from random_tensor")
print(f"Max pool tensor shape: {max_pool_tensor.shape}")
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)
##########################

# Training and testing for gpu #
torch.manual_seed(42)
train_time_start_model_2 = timer()
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_dataloader, model=model_2, loss_fn=loss_fn, optimizer=optimizer, accuracy_fn=accuracy_fn, device=device)
    test_step(data_loader=test_dataloader, model=model_2, loss_fn=loss_fn, accuracy_fn=accuracy_fn, device=device)

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_2, end=train_time_end_model_2, device=device)
model_2_results = eval_model(model=model_2, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_fn=accuracy_fn)
print(model_2_results)
################################

device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

if image_path.is_dir():

    print(f"{image_path} directory exists.")

else:

    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        
        print("Unzipping pizza, steak, sushi data...") 
        zip_ref.extractall(image_path)

def walk_through_dir(dir_path):

  for dirpath, dirnames, filenames in os.walk(dir_path):

    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

walk_through_dir(image_path)

train_dir = image_path / "train"
test_dir = image_path / "test"
print(train_dir, test_dir)

random.seed(42)
image_path_list = list(image_path.glob("*/*/*.jpg"))
random_image_path = random.choice(image_path_list)
image_class = random_image_path.parent.stem
img = Image.open(random_image_path)
img_as_array = np.asarray(img)

# plt.figure(figsize=(10, 7))
# plt.imshow(img_as_array)
# plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
# plt.axis(False)
# plt.show()

data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

def plot_transformed_images(image_paths, transform, n=3, seed=42):

    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)

    for image_path in random_image_paths:
        
        with Image.open(image_path) as f:
            
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")
            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
    plt.show()

# plot_transformed_images(image_path_list, 
#                         transform=data_transform, 
#                         n=3)

train_data = datasets.ImageFolder(root=train_dir, transform=data_transform, target_transform=None)
test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)
print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
img, label = train_data[0][0], train_data[0][1]
img_permute = img.permute(1, 2, 0)
print(f"Original shape: {img.shape} -> [color_channels, height, width]")
print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")
class_names = train_data.classes

# plt.figure(figsize=(10, 7))
# plt.imshow(img.permute(1, 2, 0))
# plt.axis("off")
# plt.title(class_names[label], fontsize=14)
# plt.show()

train_dataloader = DataLoader(dataset=train_data, batch_size=1, num_workers=1, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=1, num_workers=1, shuffle=False)
print(train_dataloader, test_dataloader)
img, label = next(iter(train_dataloader))
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:

        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

print(find_classes(train_dir))

class ImageFolderCustom(Dataset):
    
    def __init__(self, targ_dir: str, transform=None) -> None:

        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg")) 
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name 
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx 
        else:
            return img, class_idx 

train_transforms = transforms.Compose([transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()])
test_transforms = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

train_data_custom = ImageFolderCustom(targ_dir=train_dir, transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)
print(train_data_custom, test_data_custom)

train_dataloader_custom = DataLoader(dataset=train_data_custom, batch_size=1, num_workers=0, shuffle=True)
test_dataloader_custom = DataLoader(dataset=test_data_custom, batch_size=1, num_workers=0, shuffle=False) 
print(train_dataloader_custom, test_dataloader_custom)

train_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.TrivialAugmentWide(num_magnitude_bins=31), transforms.ToTensor()])
test_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

simple_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")
train_dataloader_simple = DataLoader(train_data_simple, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataloader_simple = DataLoader(test_data_simple, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
train_dataloader_simple, test_dataloader_simple

class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv_block_2 = nn.Sequential(nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
                                          nn.ReLU(),
                                          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
                                          nn.ReLU(),
                                          nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=hidden_units*16*16,out_features=output_shape))
    
    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_data.classes)).to(device)
print(model_0)

img_batch, label_batch = next(iter(train_dataloader_simple))
img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
print(f"Single image shape: {img_single.shape}\n")
model_0.eval()

with torch.inference_mode():
    pred = model_0(img_single.to(device))

print(f"Output logits:\n{pred}\n")
print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
print(f"Actual label:\n{label_single}")
summary(model_0, input_size=[1, 3, 64, 64])

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):

    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module):

    model.eval() 
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        
        for batch, (X, y) in enumerate(dataloader):
           
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module = nn.CrossEntropyLoss(), epochs: int = 5):
    
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):

        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer)
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results

torch.manual_seed(42) 
torch.cuda.manual_seed(42)
NUM_EPOCHS = 5
model_0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_data.classes)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)
start_time = timer()
model_0_results = train(model=model_0, train_dataloader=train_dataloader_simple, test_dataloader=test_dataloader_simple, optimizer=optimizer,
                        loss_fn=loss_fn, epochs=NUM_EPOCHS)
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
print(model_0_results.keys())

def plot_loss_curves(results: Dict[str, List[float]]):
    loss = results['train_loss']
    test_loss = results['test_loss']
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']
    epochs = range(len(results['train_loss']))
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

# plot_loss_curves(model_0_results)
# plt.show()

train_transform_trivial_augment = transforms.Compose([transforms.Resize((64, 64)), transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                                      transforms.ToTensor()])

test_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform_trivial_augment)
test_data_simple = datasets.ImageFolder(test_dir, transform=test_transform)

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

torch.manual_seed(42)
train_dataloader_augmented = DataLoader(train_data_augmented, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_dataloader_simple = DataLoader(test_data_simple, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
torch.manual_seed(42)
model_1 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(train_data_augmented.classes)).to(device)
print(model_1)

torch.manual_seed(42) 
torch.cuda.manual_seed(42)
NUM_EPOCHS = 5
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)
start_time = timer()
model_1_results = train(model=model_1, train_dataloader=train_dataloader_augmented, test_dataloader=test_dataloader_simple, optimizer=optimizer,
                        loss_fn=loss_fn, epochs=NUM_EPOCHS)
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")
custom_image_path = data_path / "04-pizza-dad.jpeg"
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)
custom_image = custom_image / 255. 
print(f"Custom image tensor:\n{custom_image}\n")
print(f"Custom image shape: {custom_image.shape}\n")
print(f"Custom image dtype: {custom_image.dtype}")
plt.imshow(custom_image.permute(1, 2, 0)) # need to permute image dimensions from CHW -> HWC otherwise matplotlib will error
plt.title(f"Image shape: {custom_image.shape}")
plt.axis(False);
plt.show()

custom_image_transform = transforms.Compose([transforms.Resize((64, 64))])
custom_image_transformed = custom_image_transform(custom_image)

model_1.eval()

with torch.inference_mode():
    
    custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)
    print(f"Custom image transformed shape: {custom_image_transformed.shape}")
    print(f"Unsqueezed custom image shape: {custom_image_transformed_with_batch_size.shape}")
    custom_image_pred = model_1(custom_image_transformed.unsqueeze(dim=0).to(device))

print(custom_image_pred)
print(f"Prediction logits: {custom_image_pred}")
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
print(f"Prediction probabilities: {custom_image_pred_probs}")
custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
print(f"Prediction label: {custom_image_pred_label}")
custom_image_pred_class = class_names[custom_image_pred_label.cpu()] 
print(custom_image_pred_class)

def pred_and_plot_image(model: torch.nn.Module, image_path: str, class_names: List[str] = None, transform=None, device: torch.device = device):

    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    target_image = target_image / 255. 

    if transform:
        target_image = transform(target_image)

    model.to(device)
    model.eval()

    with torch.inference_mode():

        target_image = target_image.unsqueeze(dim=0)
        target_image_pred = model(target_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    plt.imshow(target_image.squeeze().permute(1, 2, 0))

    if class_names:

        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    
    else: 
    
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    
    plt.title(title)
    plt.axis(False)

# pred_and_plot_image(model=model_1, image_path=custom_image_path, class_names=class_names, transform=custom_image_transform, device=device)
# plt.show()