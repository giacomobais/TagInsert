import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import yaml
from models.Prange.model.CCGNode import CCGNode
from models.Prange.scripts.preprocess import create_mapping

def load_config():
    """The function loads the config from the config.yaml file"""
    with open('models/Prange/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

"""# Model + Encoder"""

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, sequence_lengths, tgt, mode):
        "Take in and process masked src and target sequences."
        encodings = self.encoder(src)

        out = self.decode(encodings, sequence_lengths, tgt, mode)
        return out

    def encode(self, src):
        return self.encoder(src)

    def decode(self, memory, sequence_lengths, tgt, mode):
        out = self.decoder(memory, sequence_lengths, tgt, mode)
        return out


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class BERT_Encoder(nn.Module):
    def __init__(self, bert_model, tokenizer):
        super(BERT_Encoder, self).__init__()
        self.bert_encoder = bert_model
        self.tokenizer = tokenizer
        self.config = load_config()


    def forward(self, original_sentences):
        marked_text = [" ".join(sentence) for sentence in  original_sentences]
        # use tokenizer to get tokenized text and pad up to 100
        tokens_tensor = [self.tokenizer(text, padding="max_length", truncation=True, max_length=self.config['model']['BLOCK_SIZE_BERT'], return_tensors="pt") for text in marked_text]
        attention_mask = torch.stack([t['attention_mask'] for t in tokens_tensor])
        tokens_tensor = torch.stack([t['input_ids'] for t in tokens_tensor]).squeeze(1)
        mappings = [create_mapping(sentence, tokens_tensor[i], cased = True, subword_manage='prefix', tokenizer = self.tokenizer) for i, sentence in enumerate(original_sentences)]
        tokens_tensor = tokens_tensor.to(self.config['model']['device'])
        attention_mask = attention_mask.to(self.config['model']['device'])
        self.bert_encoder = self.bert_encoder.to(self.config['model']['device'])
        outputs = self.bert_encoder(tokens_tensor, attention_mask = attention_mask)
        last_hidden_state = outputs.last_hidden_state
        encodings = torch.zeros((len(original_sentences), self.config['model']['BLOCK_SIZE'], self.config['model']['d_model']), requires_grad = True).to(self.config['model']['device'])
        for i, sentence in enumerate(mappings):
            for j, word in enumerate(sentence):
                if word is None:
                    continue
                encodings[i][word] = last_hidden_state[i][j]
        return encodings

"""# AddrMLP"""

class Decoder(nn.Module):

    def __init__(self, tgt_vocab, enc_to_idx, atomic_to_idx):
        super(Decoder, self).__init__()
        self.config = load_config()
        self.dropout = nn.Dropout(0.2)
        self.tgt_vocab = tgt_vocab
        self.enc_to_idx = enc_to_idx
        self.atomic_to_idx = atomic_to_idx
        self.layernorm = nn.LayerNorm(self.config['model']['d_model'])
        self.feats = nn.Sequential(
            nn.Linear(self.config['model']['d_model'], self.config['model']['d_model']),
            nn.GELU(),
            self.dropout
        )
        self.mlp = nn.Sequential(
            self.layernorm,
            nn.Linear(self.config['model']['d_model'], self.config['model']['d_model']),
            nn.GELU(),
            self.dropout,
            self.layernorm,
            nn.Linear(self.config['model']['d_model'], self.config['model']['d_model']),
            nn.GELU(),
            self.dropout
        )
        self.softmax_layer = nn.Sequential(
            nn.Linear(self.config['model']['d_model'], self.tgt_vocab, bias=False),
            nn.Softmax(dim=-1)
        )
        self.feat_embs = nn.Embedding(num_embeddings=len(self.enc_to_idx), embedding_dim=self.config['model']['d_model'])
        

    def predict_data(self, node, memory, current_depth):
        pos = node.pos
        if len(pos) == 0:
            pos = '0'
        else:
            pos = ''.join(map(str, pos))
        slashes = node.get_ancestors_slashes(self.atomic_to_idx)
        if len(slashes) == 0:
            slashes = '0'
        else:
            slashes =  ''.join(map(str, slashes))
        features = torch.tensor(self.enc_to_idx[pos+slashes], requires_grad = False).to(self.config['model']['device'])
        out = self.feat_embs(features)
        out = self.feats(out)
        out = out + memory
        out = self.mlp(out)
        out = self.softmax_layer(out)
        pred = torch.argmax(out)
        return pred, out


    def build_tree_depth(self, root, memory, tgt, mode, current_depth):
        nodes_at_d = [root]
        logits = torch.zeros((2**(self.config['model']['MAX_DEPTH']+1)-1, self.tgt_vocab)).to(self.config['model']['device'])
        atomic, distribution = self.predict_data(root, memory, 0)
        logits[0] = distribution
        root.data = atomic.item()
        if mode == "train":
            label = tgt[0].item()
            root.teacher_data = label
        else:
            label = root.data
            root.teacher_data = root.data
        if label == self.atomic_to_idx['/'] or label == self.atomic_to_idx['\\']:
            for d in range(self.config['model']['MAX_DEPTH']):
                new_nodes_at_d = [None] * (2**(d+1))
                for n, node in enumerate(nodes_at_d):
                    if node is None:
                        continue
                    if mode == "train":
                        label = node.teacher_data
                    else:
                        label = node.data

                    # print(f'I predicted {node.data} for the node number {n} at depth {d}, but it was actually {node.teacher_data}.')
                    if label == self.atomic_to_idx['/'] or label == self.atomic_to_idx['\\']:
                        left_child = CCGNode(supertag = None)
                        right_child = CCGNode(supertag = None)

                        node.add_children(left_child, right_child)
                        left_atomic, left_distribution = self.predict_data(left_child, memory, d+1)
                        right_atomic, right_distribution = self.predict_data(right_child, memory, d+1)
                        left_child.data = left_atomic.item()
                        right_child.data = right_atomic.item()
                        new_nodes_at_d[n*2] = left_child
                        new_nodes_at_d[(n*2)+1] = right_child
                        logits[(2**(d+1)-1) + n*2] = left_distribution
                        logits[(2**(d+1)-1) + (n*2)+1] = right_distribution
                        if mode == 'train':
                            left_child.teacher_data = tgt[(2**(d+1)-1) + n*2].item()
                            right_child.teacher_data = tgt[(2**(d+1)-1) + (n*2)+1].item()
                        else:
                            left_child.teacher_data = left_child.data
                            right_child.teacher_data = right_child.data
                        # print(f'Then I predicted {left_child.data} for the left child number {n} at depth {d} and {right_child.data} for the right child at depth {d}. Correct were {left_child.teacher_data} and {right_child.teacher_data}')
                for n, node in enumerate(new_nodes_at_d):
                    if node is None:
                        continue
                    flag = 0
                    if node.teacher_data == self.atomic_to_idx['/'] or node.teacher_data == self.atomic_to_idx['\\']:
                        flag = 1
                        break
                if flag == 0:
                    break

                nodes_at_d = new_nodes_at_d
        # print('--------------')
        return root, logits


    def forward(self, memory, seq_length, tgt, mode):
        # memory has shape (batch_size, seq_len, hidden_dim)
        out = []
        logits = torch.zeros((memory.size(0), self.config['model']['BLOCK_SIZE'], 2**(self.config['model']['MAX_DEPTH']+1)-1, self.tgt_vocab)).to(self.config['model']['device'])
        # print(memory)
        for i, sequence in enumerate(memory):
            sequence_trees = []
            for j in range(seq_length[i]):
                # print(f'doing word number {j}')
                CCG_tree = CCGNode(supertag = None)
                if mode == 'train':
                    CCG_tree, tag_logits = self.build_tree_depth(CCG_tree, memory[i, j], tgt[i, j], mode, current_depth=0)
                else:
                    CCG_tree, tag_logits = self.build_tree_depth(CCG_tree, memory[i, j], None, mode, current_depth=0)
                sequence_trees.append(CCG_tree)
                logits[i][j] = tag_logits
            out.append(sequence_trees.copy())
        return out, logits
