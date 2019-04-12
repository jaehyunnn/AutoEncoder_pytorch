# Auto-Encoders for MNIST

PyTorch implementation of Auto-Encoders 

**Schematic structure of an auto-encoder with 3 fully connected hidden layers:** 

![](https://upload.wikimedia.org/wikipedia/commons/2/28/Autoencoder_structure.png)

## Dependencies

- Python 3 (anaconda)
- PyTorch 1.0.0, torchvision

## Usage

#### Train
```
python train.py \
--model <basic/conv/denoise> \
--lr <learning rate> \
--num_epoch <maximum epoch> \
--batch_size <batch size>
```
#### Test
```
python test.py \
--model <basic/conv/denoise> \
--input_path <test image path> \
--checkpoint_path <checkpoint path>
```

## TODO

- [x] Implement 'Basic-AE'
- [x] Implement 'Convolutional-AE'
- [ ] Implement 'Denoising-AE'

## Author

[@ Jae-Hyun Park](https://github.com/jaehyunnn)