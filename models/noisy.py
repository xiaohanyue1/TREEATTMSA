import torch
import torch.nn as nn


class CreatZ(nn.Module):
    def __init__(self, config):
        super(CreatZ, self).__init__()
        self.mu = nn.Sequential(
            # nn.BatchNorm1d(args.hidden_sz, eps=2e-5,affine=False),
            # nn.Dropout(p=0.4),
            # Flatten(),
            nn.Linear(50*256*2, 768))
        # nn.BatchNorm1d(128,eps=2e-5))
        self.logvar = nn.Sequential(
            # nn.BatchNorm1d(args.hidden_sz, eps=2e-5,affine=False),
            # nn.Dropout(p=0.4),
            # Flatten(),
            nn.Linear(50*256*2, 768))
        # nn.BatchNorm1d(128,eps=2e-5))
        # self.clf.apply(self.bert.init_bert_weights)
        self.clf = nn.Linear(768, 1)

    def forward(self, x):
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        mu = self.mu(x)  # batch_size*200
        logvar = self.logvar(x)  # batch_size*200
        x = self._reparameterize(mu, logvar)
        out = self.clf(x)
        return mu, logvar, out

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        sampler = epsilon * std
        return mu + sampler
