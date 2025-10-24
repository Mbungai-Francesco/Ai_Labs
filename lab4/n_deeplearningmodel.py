import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        ## Convolutional layers, where weights represent conv kernels
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        # 6 input channels (the output of the last layer), 16 output channels, 3x3 square convolution
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)

        ## Linear layer: MLP, i.e. fully-connected layer.
        self.fc1 = nn.Linear(in_features = 16 * 6 * 6, out_features = 120)  # 6*6 from the image dimension, and 16 for the number of channels
        self.fc2 = nn.Linear(in_features = 120, out_features = 84) # 120 is output of the previous layer.
        self.fc3 = nn.Linear(in_features = 84, out_features = 10) # 84 is the output of the previous layer, 10 is the number of classes.

    def forward(self, x):
        # Conv1, then max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # Conv2, then max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x)) # Reshape each image, processed by conv, into a vector (required for linear layers)
        # 1st Linear layer
        x = F.relu(self.fc1(x))
        # 2nd Linear layer
        x = F.relu(self.fc2(x))
        # 3rd Linear layer
        x = self.fc3(x)
        return x

    def num_flat_features(self, x): # Computes the number of flat (*"vectorized"*) features from a 2D conv.
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

testinput = torch.randn(1, 1, 33, 33) # Batch dim, number of channels, height, width
out = net(testinput)
print(out)
print("Shape of the ouput: ", out.shape)

# first layer of the model, as defined in the forward function, but we call the conv1 module from the model definition
x = testinput
print(f"Initial shape of the input : {x.shape}")
x = (F.relu(net.conv1(x)))
print(f"Shape after the first convolutional layer (conv1, relu) : {x.shape}")

x = F.max_pool2d(x, (2, 2))

print(f"Shape after max pooling (max_pool2d) with a 2x2 window: {x.shape}")
# Second layer 
x = F.max_pool2d(F.relu(net.conv2(x)), 2)

print(f"Shape after the second convolutional layer and 2x2 max pool (conv2, relu, max_pool2d) : {x.shape}")

x = x.view(-1, net.num_flat_features(x))
print(f"Shape after reshaping (flattening to a 1D vector) : {x.shape}")

x = F.relu(net.fc1(x))
print(f"Shape after FC1 : {x.shape}")

x = F.relu(net.fc2(x))
print(f"Shape after FC2 : {x.shape}")

x = net.fc3(x)
print(f"Shape after FC3, output of the model : {x.shape}")

net.zero_grad()

### LOSS FUNCTION###
output = net(testinput)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

### BACKPROPAGATION ###
net.zero_grad() # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad) # look at gradients for conv1

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
    
import torch.optim as optim

# Create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01) 

# In your training loop:
optimizer.zero_grad()   # Zeroes the gradient buffers
output = net(testinput) # Makes the prediction
loss = criterion(output, target) # Computes the loss
loss.backward() # Computes the gradient
optimizer.step() # Does the update (Gradient Descent)