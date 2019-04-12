from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dims, code_dims, use_cuda=True):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dims, 64, kernel_size=(3,3), padding=1), # b x 64 x 28 x 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)), # b x 64 x 14 x 14
            nn.Conv2d(64, code_dims, kernel_size=(3,3), padding=1), # b x code_dims x 14 x 14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)) # b x code_dims x 7 x 7
        )

        # Use the deconvolutional layer as a decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(code_dims, 64, kernel_size=(3,3), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, input_dims, kernel_size=(4,4), stride=2),
            nn.Tanh()
        )
        if use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()

    def encoding(self, x):
        return self.encoder(x)

    def decoding(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoding(x)
        x = self.decoding(x)
        return x