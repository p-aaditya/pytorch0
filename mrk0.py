import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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
	plt.show()

#plot_predictions()

class LinearRegressionModel(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        self.weights = torch.nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)
        self.bias = torch.nn.Parameter(torch.randn(1, dtype=torch.float),requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

torch.manual_seed(42)
model_0 = LinearRegressionModel()
print(list(model_0.parameters()))
print(model_0.state_dict())

with torch.inference_mode():
	y_preds = model_0(x_test)

print(f"Number of testing samples: {len(x_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

plot_predictions(predictions=y_preds)
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.01)

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

plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


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
    plt.show()

plot_predictions()

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
model_1.to(device)

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
plot_predictions(predictions=y_preds.cpu())

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