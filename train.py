from __future__ import print_function, division
import argparse
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from model.basic_AE import AutoEncoder

print('\nAutoEncoder training script')

# Argument parsing
parser = argparse.ArgumentParser(description='AutoEncoder PyTorch implementation')
# Hyper parameters
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum constant')
parser.add_argument('--num-epochs', type=int, default=1000, help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=10000, help='training batch size')
parser.add_argument('--weight-decay', type=float, default=0, help='weight decay constant')
parser.add_argument('--seed', type=int, default=1, help='Pseudo-RNG seed')
# Reload model flag
parser.add_argument('--load-model', type=bool, default=False, help='loading the trained model checkpoint')

args = parser.parse_args()

# Cuda check
use_cuda = torch.cuda.is_available()

# Seed
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

# MNIST dataset
dataset = MNIST(root='./datasets',
                train=True,
                transform=transforms.ToTensor(),
                download=True)

# Data loader
data_loader = DataLoader(dataset=dataset,
                         batch_size=args.batch_size,
                         shuffle=True,
                         num_workers=8)

# Create model
print('Creating model...')
model = AutoEncoder(input_size=28*28, code_size=32, use_cuda=use_cuda)
if use_cuda:
    model = model.cuda()

# Loss function & Optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# Train section
print('Starting training...\n')
model.train()
train_loss = 0
for epoch in range(args.num_epochs):
    for batch in data_loader:
        # Sample batch
        x, _ = batch
        x = x.view(x.size(0), -1) # flatten
        x = Variable(x).cuda() if use_cuda else Variable(x)
        # Forward
        x_prime = model(x)
        loss = criterion(x_prime, x)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data.cpu().numpy()
    # Print train log
    train_loss /= len(data_loader)
    if epoch % 1 == 0:
        print('EPOCH[{:3d}] Train set: Average loss: {:.4f}'.format(epoch+1,train_loss))

# Save trained model
save_path = 'trained_models/checkpoint.pth.tar'
torch.save(model.state_dict(), save_path)
print("Saved trained checkpoint ---- ",save_path)

