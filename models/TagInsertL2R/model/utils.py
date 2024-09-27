import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from models.TagInsertL2R.model.model import make_model
import yaml

def subsequent_mask(size):
    """
    Mask out subsequent positions.
    """
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, embs = None, pad=0):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        self.embs = embs
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss
    
def save_model(model, optimizer, lr_scheduler, train_losses, val_losses, epochs, path):
    """
    Save the model, optimizer, lr_scheduler, train_losses, val_losses, and trained epochs to a file
    """
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler' : lr_scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs': epochs
                }, path)
  
def load_model(path, word_to_idx, POS_to_idx, config, device):
    """
    Load the model, optimizer, lr_scheduler, train_losses, val_losses, and trained epochs from a file
    """
    model = make_model(len(word_to_idx), len(POS_to_idx), d_model = config['model']['d_model'], N=config['model']['n_heads'])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['lr'], betas=(config['model']['beta1'], config['model']['beta2']), eps=config['model']['eps'])
    lr_scheduler = LambdaLR(optimizer=optimizer,lr_lambda=lambda step: rate(step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=config['model']['warmup']),)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'], strict = True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    epochs = checkpoint['epochs']
    logs = {'train_losses': train_losses, 'val_losses': val_losses, 'epochs': epochs}
    return model, optimizer, lr_scheduler, logs

def load_config():
    """
    Load the config from the config.yaml file
    """
    with open('models/TagInsertL2R/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def greedy_decode(model, src, src_mask, embs, max_len, start_symbol):
    """
    Greedy decode function for inference
    """
    # encode the source sentence (words)
    memory = model.encode(src, src_mask, embs)
    # initialize the target sentence (POS tags) with the start symbol
    ys = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)
    # decode the target sentence (POS tags)
    for i in range(max_len - 1):
        out = model.decode(
            memory, None, embs, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # get the probability distribution over the target sentence (POS tags)
        prob = model.generator(out[:, -1])
        # get the next POS tag
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.view(src.size(0), 1)
        # append the next POS tag to the target sentence (POS tags)
        ys = torch.cat((ys, next_word), dim=1)
    return ys