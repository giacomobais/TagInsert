import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import yaml
from models.Prange.model.model import EncoderDecoder, BERT_Encoder, Decoder
import torch.nn as nn
import torch
import numpy as np
from transformers import RobertaTokenizer
def load_config():
    """The function loads the config from the config.yaml file"""
    with open('models/Prange/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def make_model(tgt_vocab, enc_to_idx, atomic_to_idx, bert_model, tokenizer):
    "Helper: Construct a model from hyperparameters."
    model = EncoderDecoder(
        BERT_Encoder(bert_model, tokenizer),
        Decoder(tgt_vocab, enc_to_idx, atomic_to_idx)
    )

    # Initialize parameters with Glorot / fan_avg.
    for name, p in model.named_parameters():
        if p.dim() > 1 and not name.startswith('encoder'):
            nn.init.xavier_uniform_(p)
    return model


    
class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, criterion):
        self.criterion = criterion
        self.config = load_config()

    def __call__(self, x, y, sequence_lengths, norm):
        batch_loss = torch.zeros(self.config['model']['BATCH_SIZE'])
        # for each sentence in the batch
        for i, sentence in enumerate(x):
            words_loss = torch.zeros(sequence_lengths[i])
            # for each word in the sentence we have a tree
            for j in range(sequence_lengths[i]):
                # print(y[i][j], 'gold tree')
                # pretty_print_tree(sentence[j])
                # print('predicted supertag')
                # for each node in the tree
                # number_of_gold = torch.nonzero(y[i][j] ==CCG_to_idx['<PAD>'], as_tuple = False)[0,0].item()
                nodes_loss = torch.zeros(sentence[j].size(0))
                for n in range(sentence[j].size(0)):
                    gold_node = y[i][j][n]
                    if gold_node == -1:
                        continue

                    preds = sentence[j][n]
                    gold_node = y[i][j][n]
                    sloss = -torch.log(preds[gold_node.item()])
                    # print(sloss, 'loss for this node')
                    nodes_loss[n] = sloss
                words_loss[j] = torch.sum(nodes_loss)
                # print('--------------------------')
            # print(words_loss, 'losses for all the words')
            sentence_loss = torch.sum(words_loss)
            batch_loss[i] = sentence_loss
        final_loss = torch.sum(batch_loss) / norm
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
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

def save_model(model, optimizer, lr_scheduler, train_losses, val_losses, epochs, path):
    torch.save({
                'model_state_dict': model.state_dict(),
                'bert': model.encoder.bert_encoder,
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler' : lr_scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs': epochs
                }, path)

def load_model(path, atomic_to_idx, config, train_data, encoder_to_idx):
    checkpoint = torch.load(path)
    loaded_bert = checkpoint['bert']
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = make_model(len(atomic_to_idx), encoder_to_idx, atomic_to_idx, loaded_bert, tokenizer)
    model = model.to(config['model']['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['model']['lr'], betas=(config['model']['beta1'], config['model']['beta2']), eps=config['model']['eps'])
    # lr_scheduler = LambdaLR(optimizer=optimizer,lr_lambda=lambda step: rate(step, model_size=EMBEDDING_DIM, factor=1.0, warmup=400),)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer = optimizer,
                                                  max_lr=config['model']['lr'], epochs = config['model']['EPOCHS'], steps_per_epoch=len(train_data) // config['model']['BATCH_SIZE'] + int(len(train_data) % config['model']['BATCH_SIZE'] != 0))
    model.load_state_dict(checkpoint['model_state_dict'], strict = True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    epochs = checkpoint['epochs']
    logs = {'train_losses': train_losses, 'val_losses': val_losses, 'epochs': epochs}
    return model, optimizer, lr_scheduler, logs


def greedy_decode(model, src, sequence_lengths, max_len):
    memory = model.encode(src)

    ys = []
    batch_supertags, logits = model.decode(
        memory, sequence_lengths, None, 'val')
    for i, sentence in enumerate(batch_supertags):
        sentence_ys = []
        for j, root in enumerate(sentence):
            sentence_ys.append(root)
        ys.append(sentence_ys)
    return ys