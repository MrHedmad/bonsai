# Bonsai

This package contains a very tiny implementation of a directed tree, where each node contains list-like data.

## Usage
```py
from bonsai import Tree

# Make a new tree
tree = Tree()

# Add new nodes with `create_node(name, parent, data)
# The first node cannot have a parent. There must only be one parentless (root)
# node in the tree.
# The function returns the UUID of the created node.
root_id = tree.create_node("root", None)
# Create a node with data
tree.create_node("node_with_data", root_id, ["some", "list-like", "data"])
```

Once the tree has some nodes, you can run manipulations on it:
- Copy, slice and dice the tree:
  - `clone()`: Return a copy of the tree.
  - `subset(node_id)`: Return a branch of the tree starting from `node_id`
  - `has_ancestor(node_id, ancestor_id)`: Return `true` if `ancestor_id` is a parent of `node_id`.
- Add or remove branches:
  - `paste(other_tree, node_id)`: Paste a tree to this tree, as a branch, from `node_id`.
  - `prune(node_id)`: Remove the branch from `node_id` (inclusive).
- Retrieve nodes:
  - `get_nodes_named(name)`: Get a list of nodes with `name`.
  - `get_one_node_named(name)`: Get a single node with name `name`, or die trying.
  - `leaves`: Get a list of all leaf nodes in the tree.
  - `get_parent(node_id)`: Get the parent node of `node_id`.
  - `get_paths()`: Get all paths from the root to all the leaves, as a list of tuples.u
- Update the information of existing nodes:
  - `update_data(node_id, new_data)`: Update the data of `node_id` with `new_data`.
  - `update_name(node_id, new_name)`: Update the name of `node_id` to `new_name`.
- Get information regarding the nodes:
  - `is_leaf(node_id)`: `true` if `node_id` is a leaf.
  - `depth_of(node_id)`: Get the depth of the `node_id` as an `int`.
- Save the tree to disk:
  - `to_files(out_dir)`: Save the tree as a series of folders within folders.
  - `to_adjacency_matrix(out_stream)`: Save the adjacency matrix of the tree.
  - `to_node_list(out_stream)`: Save the list of nodes of this tree, along with the data.
