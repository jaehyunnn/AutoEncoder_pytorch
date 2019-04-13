from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, input_size, code_size, use_cuda=True):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32,code_size),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_size, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, input_size),
            nn.ReLU(inplace=True)
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