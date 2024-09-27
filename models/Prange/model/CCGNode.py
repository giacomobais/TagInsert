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
    def __init__(self, supertag, parent=None, pos = []):
        count = 0
        self.parent = parent
        self.pos = pos
        self.distribution = None
        self.teacher_data = None # force teaching label
        self.config = load_config()

        if supertag == None:
            self.data = None
            self.left = None
            self.right = None
            return
        if '(' not in supertag:
            if len(supertag) == 1:
                self.data = supertag[0]
                self.left = None
                self.right = None
            else:
                self.data = supertag[1]
                left_pos = copy.deepcopy(pos)
                left_pos.append(1)
                right_pos = copy.deepcopy(pos)
                right_pos.append(-1)
                self.left = CCGNode([supertag[0]], parent=self, pos = left_pos)
                self.right = CCGNode([supertag[2]], parent=self, pos = right_pos)
        else:
            for i, element in enumerate(supertag):
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
        if self.left is None and self.right is None:
            return 0
        else:
            return max(self.left.get_depth(), self.right.get_depth()) + 1

    def get_nodes(self):
        if not self:
            return []

        result = []
        queue = [self]
        depth = 0

        while queue and depth <= self.config['model']['MAX_DEPTH']:
            level_size = len(queue)
            for _ in range(level_size):
                node = queue.pop(0)  # Pop the first element (FIFO)
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
            # Add special integer for missing node

        return result

    def to_list(self):
        if self.left is None and self.right is None:
            return [self]
        else:
            return [self] + self.left.to_list() + self.right.to_list()

    def add_children(self, left, right):
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
        if self.parent is None:
            return []
        else:
            slash_encoding = 1 if self.parent.teacher_data == atomic_to_idx['/'] else -1
            return [slash_encoding] + self.parent.get_ancestors_slashes(atomic_to_idx)

    def to_string(self):
        if self.left is None and self.right is None:
            return self.idx_to_atomic[self.data]
        else:
            return f"({self.idx_to_atomic[self.data]} {self.left.to_string()} {self.right.to_string()})"

    def is_equal(self, other):
        return self.get_nodes() == other.get_nodes()

    def to_opaque(self, atomic_to_idx, idx_to_atomic):
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
    if root.data is None:
        print('Empty tree', flush=True)
        return
    def _print_tree(node, prefix, is_left):
        if node is not None:
            _print_tree(node.right, prefix + ("│   " if is_left else "    "), False)
            print(prefix + ("└── " if is_left else "┌── ") + str(idx_to_atomic[node.data]), flush=True)
            _print_tree(node.left, prefix + ("    " if is_left else "│   "), True)

    _print_tree(root, "", True)