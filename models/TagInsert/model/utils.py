import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from models.TagInsert.model.model import make_model
import yaml
from torch.autograd import Variable
import numpy as np

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
    """ In the case of the TagInsert model, the Batch object also contains the trajectory logic"""
    """ Each trajectory is sampled uniformly at random from the set of all permutations of the sequence length and stored"""
    """Then, during training, the next_trajectory function is used to update the target with the next trajectory step"""
    
    def __init__(self, config, POS_to_idx, src, trg=None, embs = None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            # extracting length of each sentence
            first_pads = [torch.nonzero(trg[i] == 0, as_tuple = False)[0,0].item() if torch.nonzero(trg[i] == 0, as_tuple = False).numel() > 0 else config['model']['BLOCK_SIZE']-1for i in range(trg.size(0))]
            self.sequence_lengths = first_pads
            # targets to forward
            new_trg = torch.zeros((src.size(0), config['model']['BLOCK_SIZE']), dtype = torch.int64).to(config['model']['device'])
            # targets for loss computation
            new_trg_y = torch.zeros((src.size(0), config['model']['BLOCK_SIZE']), dtype = torch.int64).to(config['model']['device']) # for each slot, the tokens yet to insert
            self.trajectory = []
            # to keep track of how many trajectory steps have been made for each sentence
            self.inserted = torch.zeros(src.size(0), dtype = torch.int64)
            for i, train in enumerate(trg):
                # random trajectory sampling
                all_ixs = torch.arange(start = 1, end = first_pads[i]+1)
                permuted_ixs = torch.randperm(all_ixs.size(0))
                permuted_all_ixs = all_ixs[permuted_ixs]
                self.trajectory.append(permuted_all_ixs)
                # constructing actual y to be forwarded
                vec = torch.full((config['model']['BLOCK_SIZE'],), POS_to_idx['<UNK>'])
                targets = torch.zeros(config['model']['BLOCK_SIZE']).to(config['model']['device'])
                vec[0] = POS_to_idx['<START>']
                vec[self.sequence_lengths[i]+1] = POS_to_idx['<END>']
                vec[self.sequence_lengths[i]+2:] = POS_to_idx['<PAD>']
                new_trg[i] = vec
                ins = 0
                # constructing targets for loss computation
                for j, ix in enumerate(vec):
                    if ix == POS_to_idx['<UNK>']:
                        targets[ins] = train[j-1]
                        ins+=1
                new_trg_y[i] = targets
            self.trg = new_trg
            self.trg_y = new_trg_y
            self.trg_mask = self.make_std_mask(self.trg, pad)
            nonpads = (self.trg_y != pad).data.sum()
            self.ntokens = nonpads
            self.embs = embs

    def next_trajectory(self):
        """Function that updates the target with the next trajectory step"""
        for i, ins in enumerate(self.inserted):
            # if all tokens have been inserted, skip the sentence to save computation
            if ins >= self.sequence_lengths[i]:
                continue
            # get the next tag position to insert
            next_tag_pos = self.trajectory[i][ins].item()
            # update the target with the next tag
            self.trg[i][next_tag_pos] = self.trg_y[i][next_tag_pos-1]
            # increment the inserted counter
            self.inserted[i] += 1

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask# & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MyLoss(nn.Module):
    """Negative log likelihood loss function that computes the loss for each sentence and returns the mean loss"""
    """Each sequence loss is defined by the sum of the negative log likelihood of the tags to be inserted in positions with <UNK> tags"""
    def __init__(self,):
        super(MyLoss, self).__init__()

    def sentence_loss(self, logits, forwarded_trgs, targets, sequence_length, POS_to_idx, config):
        """Computes the loss for a single sentence"""
        losses = torch.zeros(len(targets)).to(config['model']['device'])
        # calculating the loss for each <UNK> tag in the forwarded target
        for i, ix in enumerate(forwarded_trgs):
            # if a <PAD> is encountered, the sequence has ended
            if ix == POS_to_idx['<PAD>']:
                break   
            if ix == POS_to_idx['<UNK>']:
                # extract the tag to insert
                tag_to_insert = targets[i-1]
                # get the logit for the tag to insert
                p = logits[i][tag_to_insert]
                # calculate the loss
                losses[i] = -torch.log(p)
        # corner case where there are no <UNK> tags in the sentence        
        if torch.sum(losses) == 0:
            # the model is taught to insert and <END> tag at the last position
            tag_to_insert = POS_to_idx['<END>']
            p = logits[sequence_length+2, tag_to_insert]
            losses[0] = -torch.log(p)
        # average the losses for each sentence
        out = torch.mean(losses)
        return out

    def forward(self, logits, forwarded_trgs, targets, sequence_lengths, inserted, config, POS_to_idx):
        """Computes the loss for a batch of sentences"""
        
        batch_losses = torch.zeros(config['model']['BATCH_SIZE']).to(config['model']['device'])
        for i, k in enumerate(targets):
            # Do not calculate loss for sentences that do not have any <UNK> tags
            if inserted[i] >= sequence_lengths[i]:
                continue
            # calculate the loss for a single sentence passing the correct logits, forwarded targets, targets, sequence length, POS to idx, and config
            batch_loss = self.sentence_loss(logits[i, :, :], forwarded_trgs[i, :], targets[i, :], sequence_lengths[i], POS_to_idx, config)
            batch_losses[i] = batch_loss
        # sum the losses for each sentence in the batch
        loss = torch.sum(batch_losses)
        return loss

class SimpleLossCompute:
    "A simple loss compute function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, forwarded_y, y, sequence_lengths, norm, inserted, config, POS_to_idx):
        # extract the probabilities for each tag for each position
        x = self.generator(x) # shape (B, S * V)
        if norm == 0:
            norm = 1
        loss = self.criterion(x, forwarded_y, y, sequence_lengths, inserted, config, POS_to_idx) / norm
        return loss.data * norm, loss
    
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
    with open('models/TagInsert/config/config.yaml', 'r') as file:
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

def pad_mask(pad, tgt):
    """
    Create a mask to hide padding
    """
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask
    return tgt_mask

def greedy_decode(model, src, src_mask, embs, max_len, seq_length, POS_to_idx, device):
    """Greedy decoding function that generates the POS tags for the sentences"""
    """The functions creates a batch of targets filled with <UNK> tags and then uses the model to decode the POS tags"""
    """At each time step, a new POS tag is inserted in the sequence and the corresponding logit is masked for future predictions"""
    """To extrapolate the position to insert in (and thus the ordering), the function takes the argmax of the logits across all positions"""
    """The function returns the POS tags and the orderings"""

    # encoder forward pass
    memory = model.encode(src, src_mask, embs)
    # build the target sequence filled with <UNK> tags to pass to the decoder
    batch_size = src.size(0)
    ys = torch.full((batch_size, max_len), POS_to_idx['<UNK>']).type_as(src.data)
    ys[:, 0] = POS_to_idx['<START>']
    for i in range(batch_size):
      ys[i][seq_length[i]+1] = POS_to_idx['<END>']
      ys[i][seq_length[i]+2:] = POS_to_idx['<PAD>']
    # keep track of the positions that have been inserted
    done_positions = torch.zeros_like(ys).to(device)
    # keep track of the orderings
    orderings = torch.zeros((batch_size, max_len), dtype = torch.int64)
    for pred in range(max_len):
        # decoder forward pass
        out = model.decode(memory, src, embs, src_mask,
                           Variable(ys),
                           Variable(pad_mask(POS_to_idx['<PAD>'], ys)))

        # extract the probabilities for each tag for each position
        prob = model.generator(out)
        # mask the probabilities for the positions that have already been inserted and the <START> and <END> tags
        prob[:, 0, :] = float('-inf')
        for i in range(batch_size):
          prob[i, seq_length[i]+1:, :] = float('-inf')
        for j, sent in enumerate(done_positions):
          for i, pos in enumerate(sent):
            if pos == 1:
              prob[j, i, :] = float('-inf')

        # get the argmax indices for the probabilities
        probs = prob.to('cpu').detach().numpy()
        result = [np.unravel_index(np.argmax(r), r.shape) for r in probs]

        # update the target sequence with the predicted POS tags
        for i, (location, tag) in enumerate(result):
          if torch.sum(done_positions[i]) != seq_length[i]:
            done_positions[i, location] = 1
            ys[i, location] = tag
            orderings[i, location-1] = pred+1

    return ys, orderings