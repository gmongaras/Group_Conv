import torch
from torch import nn
import torchvision
from torch.utils.data.dataloader import DataLoader
import math



# Now let's try this with multiple output channels and a
# subset of inputs channels
class Sparse_Conv(nn.Module):
    def __init__(self, inCh, outCh, kernel_size, sub_size, device=torch.device("cpu")):
        super(Sparse_Conv, self).__init__()
        
        assert sub_size <= inCh
        self.sub_size = sub_size
        self.inCh = inCh
        self.outCh = outCh
        self.kernel_height = kernel_size[0]
        self.kernel_width = kernel_size[1]
        
        self.convs = nn.ParameterList([nn.Conv2d(self.sub_size, 1, kernel_size) for i in range(0, outCh)])
        self.weights = torch.stack([i.weight.clone().to(device) for i in self.convs])
        self.biases = torch.stack([i.bias.clone().to(device) for i in self.convs])
        
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



        X_orig = X.clone()
            
            
            
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
    






# Our input
inCh = 4
outCh = 8
L = W = 10
BS = 4
image = torch.rand(BS, inCh, L, W)

# Create a Sparse_Conv object
SC = Sparse_Conv(inCh, outCh, (5, 5), inCh)

# Create a convolution 2d object
C = nn.Conv2d(inCh, outCh, (5, 5))

# Copy weights from SC to C
C.weight = nn.Parameter(SC.weights.squeeze().clone())
C.bias = nn.Parameter(SC.biases.squeeze().clone())

# Get the sparse and normal output
SC_out = SC(image)
C_out = C(image)

# The outputs should be the same