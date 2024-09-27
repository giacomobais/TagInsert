import random
import os
import sys

# Add the project root to sys.path #TODO: include relative imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)


from models.TagInsert.scripts.preprocess import get_bert_model, get_mapping_POS, get_mapping_words
from models.TagInsert.scripts.preprocess import data_load, prepare_POS, prepare_words, get_splits, dataload_scan
from models.TagInsert.model.model import TrainState, make_model
from models.TagInsert.model.utils import save_model, load_model, load_config, rate, SimpleLossCompute, MyLoss

import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

def run_epoch_scan(
  data_iter,
  model,
  loss_compute,
  optimizer,
  scheduler,
  POS_to_idx,
  config,
  mode="train",
  accum_iter=1,
  train_state=TrainState(),
  current_losses = []
):
  """Train a single epoch"""
  total_tokens = 0
  total_loss = 0
  tokens = 0
  n_accum = 0
  for i, batch in enumerate(data_iter):
      # calculate the maximum length of the target sequence for the whole batch
      max_len = np.max(batch.sequence_lengths) - 1
      losses = torch.zeros(max_len).to(config['model']['device'])
      # calculate the loss for each trajectory step
      for traj_step in range(max_len):
          # forward pass
          out = model.forward(batch.src, batch.trg,
                              batch.src_mask, batch.trg_mask, batch.embs)
          # calculate the loss for the trajectory step
          loss, loss_node = loss_compute(out, batch.trg, batch.trg_y, batch.sequence_lengths, batch.ntokens, batch.inserted, config, POS_to_idx)
          # backward pass
          if mode == "train" or mode == "train+log":
              # loss_node = Variable(loss_node, requires_grad = True)
              loss_node.backward()
              train_state.step += 1
              train_state.samples += batch.src.shape[0]
              train_state.tokens += batch.ntokens
              if i % accum_iter == 0:
                  optimizer.step()
                  optimizer.zero_grad(set_to_none=True)
                  n_accum += 1
                  train_state.accum_step += 1
              scheduler.step()
          norm = batch.ntokens
          if norm == 0:
              norm = 1
          losses[traj_step] = loss_node * norm
          # move to the next trajectory step
          batch.next_trajectory()
      # delete the embeddings
      del batch.embs
      # calculate the loss for the whole batch
      loss_node_backward = torch.sum(losses)
      current_losses.append(loss_node_backward.item())
      total_loss += loss_node_backward.data
      total_tokens += batch.ntokens
      tokens += batch.ntokens
      # delete the loss and the loss node
      del loss
      del loss_node
      if i % 5 == 0:
          print(f'Train loss at step {i*batch.src.shape[0]}/{(i+1)*batch.src.shape[0]}: {loss_node_backward.data.item()} in mode {mode}', flush=True)
          if i != 0 and mode == "train":
              save_model(model, optimizer, scheduler, current_losses, [], 0, "models/TagInsert/saved_models/TagInsert_POS")
  return total_loss / total_tokens, train_state, current_losses



class DummyOptimizer(torch.optim.Optimizer):
    """Dummy optimizer to avoid errors when evaluating the model"""
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    """Dummy scheduler to avoid errors when evaluating the model"""
    def step(self):
        None

def resume_training(path, word_to_idx, POS_to_idx, config, device):
    """Resume training from a pre-existent model"""
    """If no pre-existent model is found, create a new model"""
    """The function returns the model, the optimizer, the scheduler, the number of trained epochs, the training losses and the validation losses"""
    try:
        model, model_opt, lr_scheduler, logs = load_model(path, word_to_idx, POS_to_idx, config, device)
        trained_epochs = logs['epochs']
        train_losses = logs['train_losses']
        val_losses = logs['val_losses']
        print('Loading a pre-existent model.')
    except:
        model = make_model(len(word_to_idx), len(POS_to_idx), d_model = config['model']['d_model'], N=config['model']['n_heads'])
        model_opt = torch.optim.Adam(model.parameters(), lr=config['model']['lr'], betas=(config['model']['beta1'], config['model']['beta2']), eps=config['model']['eps'])
        lr_scheduler = LambdaLR(optimizer=model_opt,lr_lambda=lambda step: rate(step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=config['model']['warmup']),)
        trained_epochs = 0
        train_losses = []
        val_losses = []
        print('No pre-trained model found.')
    model.train()
    model = model.to(device)
    return model, model_opt, lr_scheduler, trained_epochs, train_losses, val_losses

def train(path,train_data, val_data, train_sentence_tokens, val_sentence_tokens, word_to_idx, POS_to_idx, config, device):
    """Train the model"""
    """The function takes the path to the pre-existent model, the training data, the validation data, the training sentence tokens, the validation sentence tokens, the word to index mapping, the POS to index mapping, the config, and the device"""
    """If no pre-existent model is found, it creates a new model"""
    """The function returns the model, the optimizer, the scheduler, the training losses, the validation losses and the number of trained epochs"""
    
    model, optimizer, lr_scheduler, trained_epochs, train_losses, val_losses = resume_training(path, word_to_idx, POS_to_idx, config, device)
    # instantiate loss function
    criterion = MyLoss()

    for epoch in range(config['model']['EPOCHS']):
        model.train()
        train_losses = run_epoch_scan(dataload_scan(train_data, train_sentence_tokens, config['model']['device'], bert_model, tokenizer, config, POS_to_idx),model,SimpleLossCompute(model.generator, criterion),optimizer,lr_scheduler, POS_to_idx, config, mode="train", current_losses = train_losses)[2]
        with torch.no_grad():
            model.eval()
            val_losses = run_epoch_scan(dataload_scan(val_data, val_sentence_tokens, config['model']['device'], bert_model, tokenizer, config, POS_to_idx),model,SimpleLossCompute(model.generator, criterion),DummyOptimizer(),DummyScheduler(), POS_to_idx, config, mode="eval", current_losses = val_losses)[2]
        trained_epochs += 1
    return model, optimizer, lr_scheduler, train_losses, val_losses, trained_epochs

if __name__ == "__main__":
    # load config for hyperparameters
    config = load_config()
    # load bert model
    bert_model, tokenizer = get_bert_model(config['model']['bert']['model_name'], config['model']['bert']['device'])

    # get data
    train_sentence_tokens, train_sentence_POS, val_sentence_tokens, val_sentence_POS, test_sentence_tokens, test_sentence_POS, vocab, vocab_POS = data_load(config['model']['train_file'], config['model']['val_file'], config['model']['test_file'])

    # get mapping
    POS_to_idx, idx_to_POS = get_mapping_POS(vocab_POS)
    word_to_idx, idx_to_word, vocab = get_mapping_words(vocab)

    # prepare POS
    train_sentence_POS_idx, val_sentence_POS_idx, test_sentence_POS_idx = prepare_POS(train_sentence_POS, val_sentence_POS, test_sentence_POS, POS_to_idx, config['model']['BLOCK_SIZE'])

    # prepare tokens
    train_sentence_tokens_idx, val_sentence_tokens_idx, test_sentence_tokens_idx = prepare_words(train_sentence_tokens, val_sentence_tokens, test_sentence_tokens, word_to_idx, config['model']['BLOCK_SIZE'])
    # get splits
    train_data, val_data, test_data = get_splits(train_sentence_tokens_idx, train_sentence_POS_idx, val_sentence_tokens_idx, val_sentence_POS_idx, test_sentence_tokens_idx, test_sentence_POS_idx)

    # train model
    print("Training model...")
    path = "models/TagInsert/saved_models/TagInsert_POS"
    model, optimizer, lr_scheduler, train_losses, val_losses, trained_epochs = train(path, train_data, val_data, train_sentence_tokens, val_sentence_tokens, word_to_idx, POS_to_idx, config, config['model']['device'])

    # save model
    save_model(model, optimizer, lr_scheduler, train_losses, val_losses, trained_epochs, path)
