from __future__ import print_function, division
import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image
import skimage.io as io
from PIL import Image
import matplotlib.pyplot as plt
from model import basic_AE
from model import convolutional_AE

# Argument parsing
parser = argparse.ArgumentParser(description='AutoEncoder PyTorch implementation')
# Model
parser.add_argument('--model', type=str, default='basic', help='basic/conv/denoise')
# Input image path
parser.add_argument('--input-path', type=str, default='datasets/test_data/img_5.jpg', help='Test image filename')
parser.add_argument('--checkpoint-path', type=str, default='trained_models/checkpoint.pth.tar', help='Trained model filename')
args = parser.parse_args()

# Load input image
img = Image.open(args.input_path, mode='r')
transform = transforms.ToTensor()
img = transform(img)
if args.model == 'basic':
    input = img.view(-1)
else:
    input = img.unsqueeze(0).unsqueeze(0)

# Create model
print('Creating model...')
if args.model == 'basic':
    model = basic_AE.AutoEncoder(input_size=28*28, code_size=32, use_cuda=False)
elif args.model == 'conv':
    model = convolutional_AE.AutoEncoder(input_dims=1, code_dims=32, use_cuda=False)

# Load trained weights
checkpoint = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint)

# Inference
model.eval()

code = model.encoding(input)
recons = model.decoding(code)
if args.model == 'conv':
    input_PIL = transforms.ToPILImage(mode='L')(input.unsqueeze(0))
    recons_PIL = transforms.ToPILImage(mode='L')(recons.unsqueeze(0))
else:
    input_PIL = transforms.ToPILImage(mode='L')(input.view(28,28).unsqueeze(0))
    recons_PIL = transforms.ToPILImage(mode='L')(recons.view(28,28).unsqueeze(0))

# Display result
fig, axs = plt.subplots(1, 2)
axs[0].imshow(input_PIL)
axs[0].set_title('Input')
axs[1].imshow(recons_PIL)
axs[1].set_title('Reconstructed')
plt.show()

# Save result
input_PIL.save('results/input.jpg')
recons_PIL.save('results/recons.jpg')



