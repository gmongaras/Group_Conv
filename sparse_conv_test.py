import torch
from torch import nn
import torchvision
from torch.utils.data.dataloader import DataLoader
import math

device = torch.device("cuda:0")



# Now let's try this with multiple output channels and a
# subset of inputs channels
class Sparse_Conv(nn.Module):
    def __init__(self, inCh, outCh, kernel_size, sub_size, device):
        super(Sparse_Conv, self).__init__()
        
        assert sub_size <= inCh
        self.sub_size = sub_size
        self.inCh = inCh
        self.outCh = outCh
        self.kernel_height = kernel_size[0]
        self.kernel_width = kernel_size[1]

        # Useed for weight initialization. Note that instead of
        # using inCh, sub_size is used.
        k = 1
        for s in kernel_size:
            k *= s
        k = 1/(sub_size*k)

        # Create the weights to be of shape (outCh, 1, inCh, kernel_height, kernel_width)
        self.weights = torch.empty(outCh, 1, sub_size, kernel_size[0], kernel_size[1], device=device)

        # Create biases of shape (outCh, 1)
        self.biases = torch.empty(outCh, 1, device=device)

        # Initialize the weights using a uniform distribtuion accoring to k
        torch.nn.init.uniform_(self.weights, a=-(k**0.5), b=k**0.5)
        torch.nn.init.uniform_(self.biases, a=-(k**0.5), b=k**0.5)

        # Register the weights as parameters
        self.weights = nn.Parameter(self.weights)
        self.biases = nn.Parameter(self.biases)
        
        # self.convs = nn.ParameterList([nn.Conv2d(self.sub_size, 1, kernel_size) for i in range(0, outCh)])
        # self.weights = torch.stack([i.weight.clone().to(device) for i in self.convs])
        # self.biases = torch.stack([i.bias.clone().to(device) for i in self.convs])
        # del self.convs
        
    def forward(self, X):
        if len(X.shape) == 3:
            X = X.unsqueeze(0)
            
            
            
        # Get the h/W output
        h = X.shape[-2] - self.kernel_height
        if self.kernel_height % 2 != 0:
            h += 1
        w = X.shape[-1] - self.kernel_width
        if self.kernel_width % 2 != 0:
            w += 1
            
            
            
        # Number of desired channels
        desired_channels = self.outCh+self.sub_size-1
        # Number of times to repeat the tensor to get to that goal
        num_repeats = math.ceil(desired_channels/self.inCh)
        # Repeat the image num_repeats times along the channels
        X = X.repeat(1, num_repeats, 1, 1)
        # Slice the rest off that we don't need
        X = X[:, :desired_channels]
        
        

            
        # Pad the image by sub_size-1 along the channels to become (N, C+sub_size-1, L, W)
        # X = torch.nn.functional.pad(input=X.unsqueeze(0), pad=(0,0,0,0,0,self.sub_size-1), mode="circular").squeeze(0)
        
        # Unfold image (batch_size, channels+sub_size-1, windows, kernel_height, kernel_width)
        X = X.unfold(2, self.kernel_height, 1).unfold(3, self.kernel_width, 1)
        X = X.contiguous().view(X.shape[0], X.shape[1], -1, self.kernel_height, self.kernel_width)

        # Let's unfold this tensor to be of shape (batch_size, outCh, windows, kernel_height, kernel_width, sub_size)
        X = X.unfold(1, self.sub_size, 1)

        # Make tensor of shape (batch_size, windows, outCh, sub_size, kernel_height, kernel_width)
        X = X.permute(0, 2, 1, 5, 3, 4)

        # Multiply the patches with the weights in order to calculate the conv (batch_size, outCh, HW)
        X = (X * self.weights.transpose(0, 1).unsqueeze(0)).sum([3, 4, 5]).permute(0, 2, 1)
        
        # Add the biases
        X += self.biases.unsqueeze(0)

        # Reshape to output shape (batch_size, outCh, H, W)
        return X.reshape(X.shape[0], -1, h, w)
    



# Model with 1x28x28 input and 10 output
class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        
        # convolution layers
        self.convs = nn.Sequential( # 1x28x28
            Sparse_Conv(1, 32, (5, 5), 1, device), # 32x24x24
            nn.ReLU(),
            
            Sparse_Conv(32, 32, (5, 5), 8, device), # 32x20x20
            nn.ReLU(),
            nn.MaxPool2d(2), # 64x10x10
            
            Sparse_Conv(32, 64, (5, 5), 8, device), # 64x6x6
            nn.ReLU(),
            nn.MaxPool2d(2), # 64x3x3
            
            nn.Flatten(1, -1), # 3*3*64
            nn.Linear(3*3*64, 256), # 256
            nn.ReLU(),
            nn.Linear(256, 10), # 10
            nn.LogSoftmax(-1)
        ).to(device)
        
    def forward(self, X):
        return self.convs(X)







# Used to transform the data to a transor
transform = torchvision.transforms.Compose(
    [
        # Transform to a tensor
        torchvision.transforms.ToTensor(),
    ]
)

# Load in MNIST
MNIST_dataset = torchvision.datasets.MNIST("./", train=True, transform=transform, download=True)

# Used to load in the dataset
data_loader = DataLoader(MNIST_dataset, batch_size=256,
        pin_memory=True, num_workers=0, 
        drop_last=False, shuffle=True
    )








# Create the model
model = Model(device)

# Optimizer
optim = torch.optim.AdamW(model.parameters())

# Loss function
loss_funct = nn.CrossEntropyLoss()

# Training loop
epochs = 10
steps = 0
for epoch in range(0, epochs):
    # Iterate over all data
    for X,labels in data_loader:
        # Send the data through the model
        y_hat = model(X.to(device))
        
        # Get the loss
        loss = loss_funct(y_hat, labels.to(device))
        
        # Backprop the loss
        loss.backward()
        
        # Update model
        optim.step()
        optim.zero_grad()
        steps += 1
    print(f"Epoch {epoch}: {loss.detach().item()}")