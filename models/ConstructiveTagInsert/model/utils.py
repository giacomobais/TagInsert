import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import yaml
from models.ConstructiveTagInsert.model.model import make_model
import torch
from torch.autograd import Variable

def load_config():
    """The function loads the config from the config.yaml file"""
    with open('models/ConstructiveTagInsert/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config



    
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, x, y, sequence_lengths, config, inserted, locations):
        batch_loss = torch.zeros(config['model']['BATCH_SIZE'])
        # for each sentence in the batch
        for i, sentence in enumerate(x):
            if inserted[i] >= sequence_lengths[i]:
                continue
            # there is exactly one tag in the sentence that is going to be built
            words_loss = torch.zeros(1)
            # for each word in the sentence we have a tree
            for j in range(1):
                # get the location of the tag in the sentence
                position = locations[i]
                # initialize the loss for the nodes in the tree
                nodes_loss = torch.zeros(sentence[j].size(0))
                # for each node in the tree
                for n in range(sentence[j].size(0)):
                    # get the gold node
                    gold_node = y[i][position-1][n]
                    # if the gold node is -1, it is a padding node, so we skip it
                    if gold_node == -1:
                        continue
                    # get the predictions for the node
                    preds = sentence[j][n]
                    # calculate the loss for the node
                    sloss = -torch.log(preds[gold_node.item()])
                    # add the loss to the nodes_loss
                    nodes_loss[n] = sloss
                # add the loss for the nodes to the words_loss
                words_loss[j] = torch.sum(nodes_loss)
            # add the loss for the words to the sentence_loss
            sentence_loss = torch.sum(words_loss)
            # add the loss for the sentence to the batch_loss
            batch_loss[i] = sentence_loss
        # calculate the final loss for the batch, normalize it by the batch size
        final_loss = torch.sum(batch_loss) / config['model']['BATCH_SIZE']
        # return the final loss
        return final_loss.data , final_loss
    
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

class DummyOptimizer(torch.optim.Optimizer):
    """ Dummy optimizer for the model. Used for evaluation. """
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    """ Dummy scheduler for the model. Used for evaluation. """
    def step(self):
        None

def save_model(model, optimizer, lr_scheduler, train_losses, val_losses, epochs, path):
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler' : lr_scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs': epochs
                }, path)

def load_model(path, word_to_idx, CCG_to_idx, config, train_data, ATOMIC_VOCAB, encoder_to_idx, atomic_to_idx):
    """The function loads the model from the checkpoint file. If there is no checkpoint file, it creates a new model."""
    model = make_model(len(word_to_idx), len(CCG_to_idx), ATOMIC_VOCAB, encoder_to_idx, atomic_to_idx, CCG_to_idx, N = 6, d_model = 768,d_ff = 768*4, h = 8)
    model = model.to(config['model']['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['model']['lr'], betas=(config['model']['beta1'], config['model']['beta2']), eps=config['model']['eps'])
    # lr_scheduler = LambdaLR(optimizer=optimizer,lr_lambda=lambda step: rate(step, model_size=EMBEDDING_DIM, factor=1.0, warmup=400),)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer = optimizer,
                                                max_lr=config['model']['lr'], epochs = config['model']['EPOCHS'], steps_per_epoch=len(train_data) // config['model']['BATCH_SIZE'] + int(len(train_data) % config['model']['BATCH_SIZE'] != 0))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'], strict = True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    epochs = checkpoint['epochs']
    logs = {'train_losses': train_losses, 'val_losses': val_losses, 'epochs': epochs}
    return model, optimizer, lr_scheduler, logs

def pad_mask(pad, tgt):
    """The function creates a mask for the target sentences. It is used to mask the padding tokens in the target sentences."""
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask
    return tgt_mask

def greedy_decode(model, src, src_mask, embs, max_len, seq_length, CCG_to_idx, config, device, atomic_to_idx, idx_to_atomic):
    """The function performs greedy decoding with the model."""
    # encode the source sentences
    memory = model.encode(src, src_mask, embs)
    # get the batch size
    batch_size = src.size(0)
    # create the target sentences and fill them with the <UNK> tokens
    ys = torch.full((batch_size, max_len), CCG_to_idx['<UNK>']).type_as(src.data)
    # set the special <START>, <END>, <PAD> tokens
    out_ys = []
    for i in range(batch_size):
        out_ys.append([0]*max_len)
        out_ys[i][0] = CCG_to_idx['<START>']
    ys[:, 0] = CCG_to_idx['<START>']
    for i in range(batch_size):
        ys[i][seq_length[i]+1] = CCG_to_idx['<END>']
        ys[i][seq_length[i]+2:] = CCG_to_idx['<PAD>']
        out_ys[i][seq_length[i]+1] = CCG_to_idx['<END>']
    # create a tensor to keep track of the done positions
    done_positions = torch.zeros_like(ys).to(device)
    # create a tensor to keep track of the orderings
    orderings = torch.zeros((batch_size, max_len), dtype = torch.int64)
    for pred in range(max_len):
        # decode the target sentences, getting the trees across the batch and their predicted locations
        roots, _, location = model.decode(memory, src, embs, src_mask,
                        Variable(ys), None,
                        Variable(pad_mask(CCG_to_idx['<PAD>'], ys)), seq_length, 'val', CCG_to_idx, None)
        # for each tree in the batch
        for i, root in enumerate(roots):
            # if the sentence is not done
            if torch.sum(done_positions[i]) != seq_length[i]:
                # mark the predicted position as done
                done_positions[i, location[i]] = 1
                # get the opaque root
                opaque_root = root.to_opaque(atomic_to_idx, idx_to_atomic)
                # if the supertag is not OOV, insert it in the target sentence and the output sentence
                if opaque_root in CCG_to_idx:
                    ys[i, location[i]] = CCG_to_idx[opaque_root]
                    out_ys[i][location[i]] = CCG_to_idx[opaque_root]
                # if the supertag is OOV, insert the <UNK> token in the target sentence and the output sentence
                # This is a limitation of the model, as it is not able to insert OOV supertags.
                else:
                    ys[i, location[i]] =  CCG_to_idx['<UNK>']
                    out_ys[i][location[i]] = CCG_to_idx['<UNK>']
                # mark the predicted position in the ordering
                orderings[i, location[i]-1] = pred+1

    return out_ys, orderings