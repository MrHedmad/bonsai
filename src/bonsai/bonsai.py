"""A tiny tree implementation"""

from __future__ import annotations

import json
import os
from collections.abc import ItemsView
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Optional, TextIO
from uuid import uuid4 as id


class IntegrityError(Exception):
    """Raised if the tree is broken after some transformation"""

    pass


class NodeNotFoundError(Exception):
    """Rased if the tree cannot find a node in its structure"""

    pass


class DoubleRootError(Exception):
    """Raised if the tree has two roots, or if two roots try to be added."""

    pass


class NodeAmbiguityError(Exception):
    """Raised if an ambiguous node is selected."""


class Node:
    """A class representing a node in the tree.

    This node has an unique (across trees) id, and a human-readable name.
    It can hold data in the form of a list of items.
    """

    def __init__(
        self,
        parent: Optional[str],
        name: Optional[str] = None,
        data: Optional[list[Any]] = None,
    ) -> None:
        """Builder for new nodes

        Args:
            parent (Optional[str]): The id of the parent node.
            name (Optional[str], optional): The name of the node. If None,
              defaults to the id of the node.
            data (list[Any], optional): List of data to store in the node.
              Defaults to None.
        """
        self.id = str(id())
        self.name = name or self.id
        self.data = data
        self.parent = parent

    def __str__(self) -> str:
        """Turn this node into its string representation

        Returns:
            str: The representation of the node.
        """
        base = f"Node '{self.name}' <{self.id}>"
        if self.data:
            base += " + data"
        if self.parent:
            base += f" parent: <{self.parent}>"

        return base

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Node):
            raise NotImplementedError("Cannot compare a Node and a non-Node")
        if self.parent != other.parent:
            return False
        if self.name != other.name:
            return False
        if self.data != other.data:
            return False
        return True


class Tree:
    def __init__(self) -> None:
        self.nodes = {}

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Tree):
            raise NotImplementedError("Cannot compare a tree and a non-tree")
        for key, value in self.nodes:
            if key not in other.nodes.keys():
                return False
            if value != other.nodes[key]:
                return False
        return True

    @property
    def empty(self) -> bool:
        """Is the tree empty?"""
        return len(self.nodes) == 0

    def all_nodes(self) -> ItemsView[str, Node]:
        """Get an iterator to all node_id: node pairs"""
        return copy(self.nodes).items()

    def create_node(
        self,
        name: Optional[str],
        parent: Optional[str],
        data: Optional[list[Any]] = None,
    ) -> str:
        """Create a new node in the tree.

        Args:
            name (Optional[str]): The name of the node. Passed to `Node()`
            parent (str): The ID of the parent node.
            data (Optional[list[Any]], optional): Data to store in the node.
              Defaults to None.

        Raises:
            DoubleRootError: If a root node (with parent = None) is trying to be
              added when another root node is already present.
            NodeNotFoundError: If the parent node to this node could not be found.

        Returns:
            str: The node ID of the newly added node.
        """
        if not parent:
            # We are trying to add a new root node
            if any([node.parent is None for node in self.nodes.values()]):
                root_node = self.root
                raise DoubleRootError(f"The tree already has a root: {root_node}")

        # I thought about adding a test to stop you from adding a non-root node
        # to an empty tree, but the if below stops you already.

        if parent and parent not in self.nodes:
            raise NodeNotFoundError(f"Cannot find parent node {parent} in current tree")

        node = Node(parent=parent, name=name, data=data)
        self.nodes[node.id] = node

        return node.id

    def clone(self) -> Tree:
        """Clone this tree to a new tree.

        Returns:
            Tree: The clone of this tree.
        """
        return deepcopy(self)

    def subset(self, node_id: str) -> Tree:
        """Get just a branch from this tree as a new tree"""
        if node_id not in self.nodes:
            raise NodeNotFoundError(f"Node {node_id} not found.")

        # If they asked for the subset on te root, we can just give them a
        # new tree.
        if self.nodes[node_id].parent is None:
            return self.clone()

        new_nodes = {}
        for node in copy(self.nodes).values():
            if node.id == node_id:
                # This is the new root node
                node.parent = None
                new_nodes[node.id] = node

            if self.has_ancestor(node.id, node_id):
                new_nodes[node.id] = node

        new_tree = Tree()
        new_tree.nodes = new_nodes
        new_tree.check_integrity()

        return new_tree

    @property
    def root(self) -> Node:
        """Return the root of the tree.

        Returns:
            Node: The root node of the tree.

        Raises:
            NodeNotFoundError: The tree is empty, so it has no root.
        """
        if self.empty:
            raise NodeNotFoundError("Tree is empty, no root can be found.")
        return copy([node for node in self.nodes.values() if node.parent is None][0])

    def has_ancestor(self, node_id: str, ancestor_node_id: str) -> bool:
        """Test if ancestor_node is a parent of node

        Args:
            node_id (str): The ID of the base node to check
            ancestor_node_id (str): The ID of the ancestor node

        Raises:
            ValueError: If either node cannot be found in the tree.

        Returns:
            bool: If the ancestor node is a parent of the base node.
        """
        if node_id not in self.nodes:
            raise NodeNotFoundError(f"Node {node_id} not found.")

        if ancestor_node_id not in self.nodes:
            raise NodeNotFoundError(f"Node {node_id} not found.")

        # There used to be a clause that returned True if the ancestor_node_id
        # was the id of the root node. But we cannot be sure that is the case
        # if the tree is invalid, so I removed the query

        parent = self.get_parent(node_id)
        # If the query node is the root node, it cannot have any ancestors.
        if parent is None:
            return False

        while parent.parent is not None:

            if parent.id == ancestor_node_id:
                return True
            parent = self.get_parent(parent.parent)

            if parent is None:
                break

        return False

    def paste(
        self,
        other_tree: Tree,
        node_id: str,
        update_data: bool = False,
    ) -> None:
        """Paste the root of another tree to a node in this tree.

        If the other tree is empty, does nothing.

        Args:
            other_tree (Tree): The other tree to paste.
            node_id (str): The node id of this tree to paste the other tree to.
            update_data (bool, optional): Should the other tree's root node data
              be copied to the node being pasted upon? Defaults to False.

        Raises:
            NodeNotFoundError: If the node to paste data on is not found.
        """
        if node_id not in self.nodes:
            raise NodeNotFoundError(f"Node {node_id} in original tree not found.")

        if other_tree.empty:
            return

        # if we need to update this node with the other tree's root node,
        # we can do it here
        other_tree_root = other_tree.root
        paste_node = self.nodes[node_id]
        if update_data:
            paste_node.data = other_tree_root.data

        # Update the parent IDs of children in the other tree to the new root
        # id
        new_nodes = {paste_node.id: paste_node}
        for node in other_tree.nodes.values():
            if node.parent == None:
                # This is the root node. We need to get rid of this
                continue
            if node.parent == other_tree_root.id:
                node.parent = node_id
            new_nodes[node.id] = node

        # Add the new tree to the current tree
        self.nodes.update(new_nodes)
        # Check the integrity of the tree, just to be sure
        self.check_integrity()

    def check_integrity(self) -> None:
        """Check the integrity of the tree.

        This will check that the tree has a root and that all nodes have a way
        to return to the root node.

        Raises:
            IntegrityError: If the tree is not valid.
        """
        # Nothing to check here, move along.
        if self.empty:
            return

        possible_values = list(self.nodes.keys())
        found_root = False
        for node in self.nodes.values():
            if node.parent is None and not found_root:
                found_root = True
                continue
            elif node.parent is None and found_root:
                raise IntegrityError("Found two roots in the tree.")

            if node.parent not in possible_values:
                raise IntegrityError(f"Node {node} failed to validate.")

        if not found_root:
            raise IntegrityError(f"There are nodes in the tree but there is no root.")

    def get_nodes_named(self, name: str) -> Optional[list[Node]]:
        """Get all nodes with a given name.

        Since node names are not unique, this returns either None if there are
        no nodes with that name, or a list with the results.

        If you need just one node, you can use `get_one_node_named` for
        stricter checks.

        Args:
            name (str): The name of the node to look for.

        Returns:
            Optional[list[Node]]: None if no nodes have that name, or a list of
              nodes with that name.
        """
        candidates = [x for x in self.nodes.values() if x.name == name]

        if len(candidates) == 0:
            return None

        return candidates

    def get_one_node_named(self, name: str) -> Node:
        """Get one and exactly one node with a certain name.

        Raises errors if this is not the case. For a more relaxed search, use
        `get_nodes_named`.

        Args:
            name (str): The name of the node to select

        Raises:
            NodeAmbiguityError: If more than one node is found with that name.
            NodeNotFoundError: If there are no nodes with that name.

        Returns:
            Node: The found node.
        """
        nodes = self.get_nodes_named(name)

        if nodes is None:
            raise NodeNotFoundError(f"No nodes with name {name}.")

        if len(nodes) > 1:
            nodes = [str(x) for x in nodes]
            raise NodeAmbiguityError(
                f"More than one node shares the same name '{name}': {nodes}"
            )

        return nodes[0]

    def is_leaf(self, node_id: str) -> bool:
        """Is the node a leaf?

        Args:
            node_id (str): The node to look for.

        Returns:
            bool: If the node has no children, so it is a leaf.

        Raises:
            NodeNotFoundError: If the node cannot be found.
        """
        if node_id not in self.nodes:
            NodeNotFoundError(f"Node {node_id} not found.")

        for node in self.nodes.values():
            if node.parent == node_id:
                return False
        return True

    def leaves(self) -> list[Node]:
        """Get a list of all leaves in this tree.

        Raises:
            NodeNotFoundError: If the tree is empty, so there are no leaves.

        Returns:
            list[Node]: The list of nodes that are leaves in this tree.
        """
        if self.empty:
            raise NodeNotFoundError("Tree is empty. There are no leaves.")

        leaves = [node for node in self.nodes.values() if self.is_leaf(node.id)]

        return leaves

    def update_data(self, node_id: str, new_data: Optional[list[Any]]) -> None:
        """Update the data of a given node.

        Args:
            node_id (str): The node ID to update
            new_data (Optional[list[Any]]): The new data to insert in the node.

        Raises:
            NodeNotFoundError: If the node to update cannot be found.
        """
        if node_id not in self.nodes:
            NodeNotFoundError(f"Node {node_id} not found.")

        self.nodes[node_id].data = new_data

    def update_name(self, node_id: str, new_name: str) -> None:
        """Update the name of a given node.

        Args:
            node_id (str): The node ID to update
            new_name (Optional[list[Any]]): The new name of the node..

        Raises:
            NodeNotFoundError: If the node to update cannot be found.
        """
        if node_id not in self.nodes:
            NodeNotFoundError(f"Node {node_id} not found.")

        self.nodes[node_id].name = new_name

    def prune(self, node_id: str) -> None:
        """Remove a node and all children of the node from the tree

        Args:
            node_id (str): The ID of the node to prune.

        Raises:
            NodeNotFoundError: If the node to prune cannot be found.
        """
        if node_id not in self.nodes:
            NodeNotFoundError(f"Node {node_id} not found.")

        self.nodes.pop(node_id)
        for id, node in self.all_nodes():
            if node.parent == node_id:
                self.prune(id)

    def get_parent(self, node_id: str) -> Optional[Node]:
        """Get the parent of a given node, or None if it is the root.

        Args:
            node_id (str): The ID of the node to find the parent of.

        Returns:
            Optional[Node]: The parent of the node, or None if the node was the
              root.

        Raises:
            NodeNotFoundError: If the node to find the parent of cannot be found.
        """
        if node_id not in self.nodes:
            NodeNotFoundError(f"Node {node_id} not found.")

        if node_id == self.root.id:
            return None

        return self.nodes[self.nodes[node_id].parent]

    def get_paths(self) -> list[tuple[str]]:
        """Get a list of paths from the root node to every other node.

        Returns:
            list[tuple[str]]: The list of paths

        Raises:
            ValueError: If the tree is empty.
        """
        if self.empty:
            raise ValueError("The tree is empty. Nothing to give the paths of.")

        paths = []

        # For every node, get the full path to the parent.
        for node in self.nodes.values():
            path = self.get_path_of(node.id)
            paths.append(path)

        return paths

    def get_path_of(self, node_id: str) -> tuple[str]:
        """Get the path to the node, as a tuple of IDs, from the root.

        Args:
            node_id (str): The node to get the path to

        Returns:
            tuple[str]: The (ordered) list of IDs from the root to the path.
        """
        root_id = self.root.id
        path = []
        current_id = node_id

        while True:
            path.append(current_id)
            if current_id == root_id:
                # This is the root node, we are at the root
                break
            # This node should have a parent
            current_node = self.get_parent(current_id)
            assert current_node is not None, "The tree seems to be broken..."
            current_id = current_node.id

        # We made the path from the leaf to the root. We need the inverse.
        path.reverse()

        return path

    def depth_of(self, node_id: str) -> int:
        """Get the number of edges to follow to get to the node from the root

        Args:
            node_id (str): The node to get the depth of

        Returns:
            int: The depth of the node.
        """
        if node_id not in self.nodes:
            NodeNotFoundError(f"Node {node_id} not found.")

        if node_id == self.root.id:
            return 0

        i = 1  # We are already one "deep".
        parent = self.get_parent(node_id)
        while parent is not None:
            parent = self.get_parent(parent.id)
            i += 1
        return i

    def get_direct_children(self, node_id: str) -> list[Node]:
        """Get a list of the direct children of this node"""
        result = []
        for _, node in self.all_nodes():
            if node.parent == node_id:
                result.append(node)

        return result

    def __str__(self) -> str:
        return f"Tree with {len(self.nodes)} nodes"

    def to_files(
        self,
        out_dir: Path,
        all_file_name: str = "all.txt",
        data_file_name: str = "data.txt",
    ):
        if not out_dir.exists():
            os.makedirs(out_dir, exist_ok=True)

        with (out_dir / all_file_name).open("w+") as stream:
            stream.writelines([f"{x}\n" for _, x in self.all_nodes()])

        paths = self.get_paths()

        for path in paths:
            names = [self.nodes[x].name for x in path]
            names.insert(0, out_dir)
            real_path = Path(*names)
            if not real_path.exists():
                os.makedirs(real_path, exist_ok=True)

            with (real_path / data_file_name).open("w+") as stream:
                data = self.nodes[path[-1]].data
                if data:
                    stream.writelines([f"{x}\n" for x in data])

    def to_adjacency_matrix(self, out_stream: TextIO):
        rows = self.all_nodes()
        cols = self.all_nodes()

        header = [""]  # the space for the rows
        header.extend([x[0] for x in cols])
        matrix = [header]  # start with the header
        # It is important that we iterate in a row > col way,
        # since we generate the matrix row-wise
        for row_id, row_node in rows:
            row = [row_id]
            for col_id, _ in cols:
                edge = 1 if row_node.parent == col_id else 0
                row.append(edge)
            matrix.append(row)

        for row in matrix:
            out_stream.write((",".join(map(str, row))) + "\n")

    def to_node_json(self, out_stream: TextIO):
        nodes = self.all_nodes()
        data = {}
        for node_id, node in nodes:
            data[node_id] = {
                "name": node.name,
                "data": node.data,
                "parent": node.parent,
            }

        json.dump(data, out_stream, indent=4)

    def to_representation(self, out_stream: TextIO, force_uuid=False):
        """Generate a tree-like string representation of this bonsai

        Shamelessly stolen and adapted from
        https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
        since I'm too stupid to do it by my own.

        :<
        """
        CONT_INDENT = "│   "
        SPLIT_MID = "├── "
        SPLIT_END = "└── "
        INDENT = "    "

        def get_repr(node: Node) -> str:
            if force_uuid:
                return node.id
            return node.name or node.id

        def print_layer(parent_node_id: str, prefix=""):
            children = self.get_direct_children(parent_node_id)

            pointers = [SPLIT_MID] * (len(children) - 1) + [SPLIT_END]
            for pointer, child in zip(pointers, children):
                yield prefix + pointer + get_repr(child)
                if not self.is_leaf(child.id):
                    extension = CONT_INDENT if pointer == SPLIT_MID else INDENT
                    yield from print_layer(child.id, prefix=prefix + extension)

        layers = list(print_layer(self.root.id))

        out_stream.write("\n".join([get_repr(self.root)] + layers))

    @staticmethod
    def from_node_json(data: dict) -> Tree:
        """Create a tree from its JSON representation

        This does the inverse of `to_node_json`.
        """
        new_nodes = {}
        for key, value in data.items():
            node = Node(
                parent=value.get("parent"),
                data=value.get("data"),
                name=value.get("name"),
            )
            node.id = key
            new_nodes[key] = node

        new = Tree()
        new.nodes = new_nodes
        new.check_integrity()

        return new

    def get_children_named(self, parent_id: str, name: str) -> list[Node]:
        """Get, as a list, all childrens of 'parent_id' named 'name'

        Args:
            parent_id: The UUID of the parent node to get the children of
            name: The name of the children nodes to find.
        """
        childrens = self.get_direct_children(parent_id)
        return _find(childrens, lambda x: x.name == name)


class IncongruencyError(Exception):
    """Raised when two trees are not identical"""

    def __init__(self, location: str, inconsistency: str):
        self.location = location
        self.inconsistency = inconsistency

    def __str__(self):
        return f"Inconsistency error at {self.location}: {self.inconsistency}"


def _which(iterable):
    """Return which items in the iterable are True"""
    enum = [i for i, x in enumerate(iterable) if x]
    if len(enum) == 1:
        return enum[0]
    return enum


def _find(items, fn):
    """Find one and only one item in a list

    The fn is a function that takes each item and returns True or False.
    """
    checks = [fn(x) for x in items]
    if sum(checks) == 1:
        return items[_which(checks)]
    elif sum(checks) > 1:
        raise RuntimeError("Found more than one matching item")
    else:
        raise RuntimeError("Item not found")


def _check_two_way_congruency(list_1, list_2):
    """Check if all items in list_1 are in list_2, and vice-versa"""
    # Could probably forgo one check and check for len but whatever
    if not all([x in list_2 for x in list_1]):
        raise RuntimeError(
            f"Did not find some items in list 1 inside list 2. Length of one: {len(list_1)} vs two {len(list_2)}"
        )
    if not all([x in list_1 for x in list_2]):
        raise RuntimeError(
            f"Did not find some items in list 2 inside list 1. Length of one: {len(list_1)} vs two {len(list_2)}"
        )
    return True


def __get_location_of(tree: Tree, node_id):
    # Wrapper to get a unique string for each location
    path = tree.get_path_of(node_id)
    names = [tree.nodes[x].name for x in path]
    spath = ""
    for name in names:
        spath += name + "///"

    return spath.strip("/")


def compare_trees(one: Tree, two: Tree):
    """Compare two trees for congruency.

    A tree is compared with another for shape, node names and node content.
    This function fails if all nodes are not named.

    Args:
        one (Tree): The first tree to be compared
        two (Tree): The second tree to be compared

    Returns:
        True, if the two trees are identical.

    Raises:
        IncongruencyError(location, inconsistency): When a difference is detected.
    """

    def compare_childrens(parent_one_id, parent_two_id):
        first_children = one.get_direct_children(parent_one_id)
        second_children = two.get_direct_children(parent_two_id)
        first_children_names = [one.nodes[x.id].name for x in first_children]
        second_children_names = [two.nodes[x.id].name for x in second_children]

        if not first_children and not second_children:
            # There are no nodes to compare - the two parents are leaves.
            # This is the safe exit condition of the recursion.
            return
        elif not first_children:
            # There are second children but not first ones. Panic!
            raise IncongruencyError(
                __get_location_of(one, parent_one_id),
                f"This node in the first tree should have children: {second_children_names}",
            )
        elif not second_children:
            # Same, but reversed
            raise IncongruencyError(
                __get_location_of(two, parent_two_id),
                f"This node in the second tree should have children: {first_children_names}",
            )

        if not len(first_children) == len(second_children):
            raise IncongruencyError(
                f"{one.nodes[parent_one_id].name} and {two.nodes[parent_two_id].name}",
                f"Different number of children, {len(first_children)} vs {len(second_children)}: {first_children_names} vs {second_children_names}",
            )

        # It does not matter which of the two lists we check - they have the same length.
        for child in first_children:
            try:
                sibling = _find(second_children, lambda x: x.name == child.name)
            except RuntimeError as e:
                raise IncongruencyError(
                    f"{__get_location_of(one, child.id)}",
                    f"could not locate sibling node: {str(e)}",
                )

            try:
                _check_two_way_congruency(child.data, sibling.data)
            except RuntimeError as e:
                raise IncongruencyError(f"{__get_location_of(one, child.id)}", str(e))

            # We are done comparing these children: they are identical.
            # now, check in turn their children.
            compare_childrens(child.id, sibling.id)

    # Get the roots of the trees and start checking
    compare_childrens(one.root.id, two.root.id)

    return True
