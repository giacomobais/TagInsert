import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from collections import deque
import copy
import yaml

def load_config():
    """The function loads the config from the config.yaml file"""
    with open('models/Prange/config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

class CCGNode():
    """ Class for a CCG Node. It includes methods to grow the tree and return leaves and nodes that can be used during training. """    
    def __init__(self, supertag, parent=None, pos = []):
        count = 0
        self.parent = parent
        self.pos = pos
        self.distribution = None
        self.teacher_data = None # force teaching label
        self.config = load_config()
        # logic to initialize the node starting from a string representation of the supertag
        if supertag == None:
            self.data = None
            self.left = None
            self.right = None
            return
        # if the supertag is a single atomic node or a binary node (i.e. N/N)
        if '(' not in supertag:
            if len(supertag) == 1:
                self.data = supertag[0]
                self.left = None
                self.right = None
            # if the supertag is a binary node
            else:
                self.data = supertag[1]
                left_pos = copy.deepcopy(pos)
                left_pos.append(1)
                right_pos = copy.deepcopy(pos)
                right_pos.append(-1)
                self.left = CCGNode([supertag[0]], parent=self, pos = left_pos)
                self.right = CCGNode([supertag[2]], parent=self, pos = right_pos)
        else:
            # if the supertag is a complex node
            for i, element in enumerate(supertag):
                # recursively grow the tree with the proper left and right nodes
                if element in ['(']:
                    count += 1
                if element in [')']:
                    count -= 1
                if count == 0:
                    self.data = supertag[i+1]
                    full_left = supertag[:i+1]
                    full_right = supertag[i+2:]
                    left_pos = copy.deepcopy(pos)
                    left_pos.append(1)
                    right_pos = copy.deepcopy(pos)
                    right_pos.append(-1)
                    if '(' in full_left:
                        self.left = CCGNode(full_left[1:-1], parent=self, pos = left_pos)
                    else:
                        self.left = CCGNode(full_left, parent=self, pos = left_pos)
                    if '(' in full_right:
                        self.right = CCGNode(full_right[1:-1], parent=self, pos = right_pos)
                    else:
                        self.right = CCGNode(full_right, parent=self, pos = right_pos)
                    break

    def get_depth(self):
        """ The function returns the depth of the tree. """
        if self.left is None and self.right is None:
            return 0
        else:
            return max(self.left.get_depth(), self.right.get_depth()) + 1

    def get_nodes(self):
        """ The function returns the nodes of the tree in a list representation. """
        if not self:
            return []

        result = []
        queue = [self]
        depth = 0
        # pop the elements from the queue in a FIFO manner
        while queue and depth <= self.config['model']['MAX_DEPTH']:
            level_size = len(queue)
            for _ in range(level_size):
                node = queue.pop(0)  
                if node == -1:
                    result.append(-1)
                    queue.append(-1)
                    queue.append(-1)
                elif node.data is not None:
                    result.append(node.data)
                    queue.append(node.left if node.left else -1)
                    queue.append(node.right if node.left else -1)
                else:
                    result.append(-1)
            depth+=1

        return result

    def to_list(self):
        """ The function returns the nodes of the tree in a list representation. """
        if self.left is None and self.right is None:
            return [self]
        else:
            return [self] + self.left.to_list() + self.right.to_list()

    def add_children(self, left, right):
        """ The function adds the left and right children to the current node. """
        assert self.left is None and self.right is None
        self.left = left
        self.right = right
        left.parent = self
        right.parent = self
        left_i = copy.deepcopy(self.pos)
        left_i.append(1)
        right_i = copy.deepcopy(self.pos)
        right_i.append(-1)
        left.pos = left_i
        right.pos = right_i

    def get_ancestors_slashes(self, atomic_to_idx):
        """ The function returns the ancestors of the current node in a list representation. """
        if self.parent is None:
            return []
        else:
            slash_encoding = 1 if self.parent.teacher_data == atomic_to_idx['/'] else -1
            return [slash_encoding] + self.parent.get_ancestors_slashes(atomic_to_idx)

    def to_string(self, idx_to_atomic):
        """ The function returns the string representation of the tree. """
        if self.left is None and self.right is None:
            return idx_to_atomic[self.data]
        else:
            return f"({idx_to_atomic[self.data]} {self.left.to_string(idx_to_atomic)} {self.right.to_string(idx_to_atomic)})"

    def is_equal(self, other):
        """ The function checks if the current tree is equal to the other tree. """
        return self.get_nodes() == other.get_nodes()
    
    def get_nodes_at_d(self, depth):
        """ The function returns the nodes at a given depth. """
        if depth == 0:
            return [self]
        else:
            left_nodes = self.left.get_nodes_at_d(depth-1) if self.left else [None] * 2**(depth-1)
            right_nodes = self.right.get_nodes_at_d(depth-1) if self.right else [None] * 2**(depth-1)
            return left_nodes + right_nodes
            
    def get_leaves(self, depth):
        """ The function returns the leaves of the tree up until the given depth. """
        leaves = []
        for d in range(depth+1):
            nodes_at_d = self.get_nodes_at_d(d)
            for node in nodes_at_d:
                if node is not None and node.left is None:
                    leaves.append(node.teacher_data)
        return leaves

    def get_leaves_right(self, depth):
        """ The function returns all the right children leaves of the tree up until the given depth. """
        leaves = []
        for d in range(depth+1):
            nodes_at_d = self.get_nodes_at_d(d)
            for node in nodes_at_d:
                if node is None:
                    continue
                if node.parent is None and node.left is None:
                    leaves.append(node.teacher_data)
                elif node is not None and node.left is None and node.parent.right == node:
                    leaves.append(node.teacher_data)
        return leaves

    def get_leaves_left(self, depth):
        """ The function returns all the left children leaves of the tree up until the given depth. """
        leaves = []
        for d in range(depth+1):
            nodes_at_d = self.get_nodes_at_d(d)
            for node in nodes_at_d:
                if node is None:
                    continue
                if node.parent is None and node.left is None:
                    leaves.append(node.teacher_data)
                elif node is not None and node.left is None and node.parent.left == node:
                    leaves.append(node.teacher_data)
        return leaves

    def to_opaque(self, atomic_to_idx, idx_to_atomic):
        """ The function returns the opaque representation of the tree. """
        if self.data != atomic_to_idx['/'] and self.data != atomic_to_idx['\\']:
            return idx_to_atomic[self.data]
        else:
            if self.left is not None and self.right is not None: # for bad trees full of slashes
                if self.left.get_depth() == 0 and self.right.get_depth() == 0:
                    return self.left.to_opaque(atomic_to_idx, idx_to_atomic) + idx_to_atomic[self.data] + self.right.to_opaque(atomic_to_idx, idx_to_atomic)
                elif self.left.get_depth() != 0 and self.right.get_depth() == 0:
                    return '(' + self.left.to_opaque(atomic_to_idx, idx_to_atomic) + ')' + idx_to_atomic[self.data] + self.right.to_opaque(atomic_to_idx, idx_to_atomic)
                elif self.left.get_depth() != 0 and self.right.get_depth() != 0:
                    return '(' + self.left.to_opaque(atomic_to_idx, idx_to_atomic) + ')' + idx_to_atomic[self.data] + '(' + self.right.to_opaque(atomic_to_idx, idx_to_atomic) + ')'
                elif self.left.get_depth() == 0 and self.right.get_depth() != 0:
                    return  self.left.to_opaque(atomic_to_idx, idx_to_atomic) + idx_to_atomic[self.data] + '(' + self.right.to_opaque(atomic_to_idx, idx_to_atomic) + ')'
            else:
                return idx_to_atomic[self.data]

def pretty_print_tree(root, idx_to_atomic):
    """ The function prints the tree in a pretty format. """
    if root.data is None:
        print('Empty tree', flush=True)
        return
    def _print_tree(node, prefix, is_left):
        if node is not None:
            _print_tree(node.right, prefix + ("│   " if is_left else "    "), False)
            print(prefix + ("└── " if is_left else "┌── ") + str(idx_to_atomic[node.data]), flush=True)
            _print_tree(node.left, prefix + ("    " if is_left else "│   "), True)
    _print_tree(root, "", True)