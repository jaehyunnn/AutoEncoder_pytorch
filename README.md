# Auto-Encoders for MNIST

An implementation of Auto-Encoders PyTorch implementation

**Schematic structure of an auto-encoder with 3 fully connected hidden layers:** 

![](https://upload.wikimedia.org/wikipedia/commons/2/28/Autoencoder_structure.png)

## Dependencies

- Python 3 (anaconda)
- PyTorch 1.0.0, torchvision

## Usage

### Train
```
python train.py \
--lr <learning rate> \
--num_epoch <maximum epoch> \
--batch_size <batch size> \
```
### Test
```
python test.py \
--input_path <test image path> \
--checkpoint_path <checkpoint path>
```

## TODO

- [x] Implement 'Basic AE'
- [ ] Implement 'Convolution-AE'
- [ ] Implement 'Denoising-AE'

## Author

Jae-Hyun Park : https://github.com/jaehyunnn
