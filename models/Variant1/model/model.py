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
from models.Variant1.model.CCGNode import CCGNode
from models.Variant1.scripts.preprocess import create_mapping

def load_config():
    """The function loads the config from the config.yaml file"""
    with open('models/Variant1/config/config.yaml', 'r') as file:
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
    """ An encoder block that includes a BERT model to finetune. """
    def __init__(self, bert_model, tokenizer):
        super(BERT_Encoder, self).__init__()
        self.bert_encoder = bert_model
        self.tokenizer = tokenizer
        self.config = load_config()


    def forward(self, original_sentences):
        """ The forward pass of the BERT encoder. """
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

""" AddrMLP """

class PaddingLayer(nn.Module):
    """ A padding layer to pad the embeddings to the desired length. """
    def __init__(self, desired_length):
        super(PaddingLayer, self).__init__()
        self.desired_length = desired_length

    def forward(self, x):
        # Calculate the amount of padding needed
        pad_length = max(0, self.desired_length - x.size(0))
        # Pad the embeddings
        if pad_length > 0:
            x = F.pad(x, (0, pad_length), mode='constant', value=0)
        return x
        
class Decoder(nn.Module):
    """ The decoder block that includes an AddrMLP model. """
    def __init__(self, tgt_vocab, enc_to_idx, atomic_to_idx):
        super(Decoder, self).__init__()
        self.config = load_config()
        self.dropout = nn.Dropout(0.2)
        self.tgt_vocab = tgt_vocab
        self.enc_to_idx = enc_to_idx
        self.atomic_to_idx = atomic_to_idx
        self.layernorm = nn.LayerNorm(self.config['model']['d_model'])
        # Linear layer to process node features
        self.feats = nn.Sequential(
            nn.Linear(self.config['model']['d_model'], self.config['model']['d_model']),
            nn.GELU(),
            self.dropout
        )
        # MLP layer leading to the softmax layer
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
        # Softmax layer to get the probabilities
        self.softmax_layer = nn.Sequential(
            nn.Linear(self.config['model']['d_model'], self.tgt_vocab, bias=False),
            nn.Softmax(dim=-1)
        )
        # Embedding layer for the position and slash features of the nodes
        self.feat_embs = nn.Embedding(num_embeddings=len(self.enc_to_idx), embedding_dim=self.config['model']['d_model'])
        # MLP layer to process the other leaves in the context window, concatenating their features and passing them through a linear layer
        self.other_leaves_layer = nn.Sequential(
                nn.Embedding(num_embeddings=len(self.atomic_to_idx), embedding_dim=32),
                nn.Flatten(0),
                PaddingLayer(32 * (2**(self.config['model']['MAX_DEPTH']-1)-1)*self.config['model']['LEAVES_CONTEXT']*2),
                nn.Linear(in_features=32 * (2**(self.config['model']['MAX_DEPTH']-1)-1)*self.config['model']['LEAVES_CONTEXT']*2, out_features=self.config['model']['d_model'])
            )

    def predict_data(self, node, memory, current_depth, other_leaves = None):
        """ The function predicts the data for a given node. """
        # extract node position and slash features
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
        # embed the features
        features = torch.tensor(self.enc_to_idx[pos+slashes], requires_grad = False).to(self.config['model']['device'])
        out = self.feat_embs(features)
        # process the features through the MLP
        out = self.feats(out)
        # add the words features to the output
        if other_leaves is None: # root prediction, no other leaves are present
                out = out + memory
        else: # depth >= 1
            out = out + memory + other_leaves 
        # process the output through the MLP
        out = self.mlp(out)
        # get the probabilities
        out = self.softmax_layer(out)
        # get the prediction
        pred = torch.argmax(out)    
        # return the prediction and the probabilities
        return pred, out

    def grab_other_leaves(self, other_trees, depth):
        """ The function grabs the leaves from the other trees at a given depth. """
        all_leaves = []
        for tree in other_trees:
            # get the leaves up to the previous depth
            list_of_leaves = tree.get_leaves(depth-1)
            all_leaves.extend(list_of_leaves)
        # prepare the leaves for the MLP
        leaves_features = torch.tensor(all_leaves, requires_grad = False).view(-1).to(self.config['model']['device'])
        # weird cases in which no leaf is present in the other trees, then return None and predict label using only the memory
        if len(leaves_features) == 0:
            return None
        # process the leaves through the MLP
        feats = self.other_leaves_layer(leaves_features)
        return feats
    
    def build_tree_parallel(self, roots, memories, tgts, mode, current_depth):
        """ The function builds the tree in a parallel top-down manner. """
        # initialize the logits tensor for the current depth
        logits = torch.zeros((len(roots), 2**current_depth, self.tgt_vocab)).to(self.config['model']['device'])
        
        seq_length = len(roots)
        # root case
        if current_depth == 0:
            # for each word in the sentence
            for i, root in enumerate(roots):
                # get the BERT encoding of the word
                memory = memories[i]
                # predict the data for the root
                atomic, distribution = self.predict_data(root, memory, current_depth)
                # store the logits for the root
                logits[i][0] = distribution
                # set the data for the root
                root.data = atomic.item()
                # if in training mode, set the teacher data for the root
                if mode == "train":
                    label = tgts[i][0].item()
                    root.teacher_data = label
                else:
                    # if in evaluation mode, set the teacher data to the predicted data
                    label = root.data
                    root.teacher_data = root.data
                # if the root is a slash, add children to the root
                if label == self.atomic_to_idx['/'] or label == self.atomic_to_idx['\\']:
                    left_child = CCGNode(supertag = None)
                    right_child = CCGNode(supertag = None)
                    root.add_children(left_child, right_child)
        # depth >= 1
        else:
            for i, root in enumerate(roots):
                # get the nodes at depth d
                nodes_at_d = root.get_nodes_at_d(current_depth)
                # if all nodes at depth d are None, the tree is fully grown, no nodes at current depth
                if all(element is None for element in nodes_at_d):
                    continue
                # get the BERT encoding for the current sentence
                memory = memories[i]
                # get the indices for the other trees respecting the context window
                left_i = max(0, i-self.config['model']['LEAVES_CONTEXT'])
                right_i = min(seq_length, i+self.config['model']['LEAVES_CONTEXT'])
                other_roots = roots[left_i:i] + roots[i+1:right_i]
                # get the other leaves encoded
                other_leaves_encoded = self.grab_other_leaves(other_roots, current_depth)
                # for each node at depth d for the current tree
                for j, node in enumerate(nodes_at_d):

                    if node is None:
                        continue
                    # predict the data for the node
                    atomic, distribution = self.predict_data(node, memory, current_depth, other_leaves_encoded)
                    # store the logits for the node 
                    logits[i][j] = distribution
                    # set the data for the node
                    node.data = atomic.item()
                    # if in training mode, set the teacher data for the node
                    if mode == "train":
                        label = tgts[i][2**(current_depth)-1 + j].item()
                        node.teacher_data = label
                    else:
                        label = node.data
                        node.teacher_data = node.data
                    # if the node is a slash, add children to the node
                    if label == self.atomic_to_idx['/'] or label == self.atomic_to_idx['\\']:
                        left_child = CCGNode(supertag = None)
                        right_child = CCGNode(supertag = None)
                        node.add_children(left_child, right_child)
        return roots, logits


    def forward(self, memory, seq_length, tgt, mode):
        """ The forward pass of the decoder. """
        out = []
        logits = torch.zeros((memory.size(0), self.config['model']['BLOCK_SIZE'], 2**(self.config['model']['MAX_DEPTH']+1)-1, self.tgt_vocab)).to(self.config['model']['device'])
        # for each sentence in the batch
        for i, sequence in enumerate(memory):
            # initialize the sequence trees
            sequence_trees = [CCGNode(supertag = None) for _ in range(seq_length[i])]
            # for each depth in the tree, we build the tree in a parallel top-down manner
            for d in range(self.config['model']['MAX_DEPTH']):
                # if in training mode, use the teacher data, otherwise use the predicted data. In eval mode there are no targets so we use the predicted data
                if mode == 'train':
                    sequence_trees, sentence_logits = self.build_tree_parallel(sequence_trees, sequence, tgt[i], mode, current_depth=d)
                else:
                    sequence_trees, sentence_logits = self.build_tree_parallel(sequence_trees, sequence, None, mode, current_depth=d)
                # store the logits for the tree
                for j, node_logit in enumerate(sentence_logits):
                    for k, logits_at_d in enumerate(node_logit):
                        # store the logits for the tree, logits are stored flat, so we need to calculate the index
                        index = 2**d-1 + k
                        logits[i][j][index] = logits_at_d
            # add the sequence trees to the output
            out.append(sequence_trees.copy())
        return out, logits
