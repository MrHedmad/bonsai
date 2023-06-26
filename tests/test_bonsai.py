from bonsai import Tree

import io


def test_new_tree():
    tree = Tree()

    assert True


def test_add_node():
    tree = Tree()

    parent = tree.create_node("node1", None)
    tree.create_node("node2", parent)

    assert len(tree.all_nodes()) == 2


def test_prune():
    # TODO: Add me!
    pass


def test_paste():
    # TODO: Add me!
    pass


def test_to_adjacency_matrix():
    tree = Tree()

    parent = tree.create_node("root", None)
    tree.create_node("root>node1", parent)
    parent = tree.create_node("root>node2", parent)
    tree.create_node("node2>node3", parent)

    stream = io.StringIO()
    tree.to_adjacency_matrix(out_stream=stream)
    stream.seek(0)

    result = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0]]

    out = stream.readlines()
    out = [x.split(",") for x in out]

    header = [x.strip() for x in out[0]]
    rownames = [x[0] for x in out]
    assert header == rownames

    parsed = []
    for row in out[1:]:
        parsed.append(list(map(int, row[1:])))

    assert result == parsed


def test_to_node_list():
    tree = Tree()

    node_ids = []

    node_ids.append(tree.create_node("root", None, ["a", "b", "c"]))
    node_ids.append(tree.create_node("node1", node_ids[0], ["d", "e"]))
    node_ids.append(tree.create_node("node2", node_ids[1]))

    stream = io.StringIO()

    tree.to_node_list(stream)
    stream.seek(0)

    result = (
        f"{node_ids[0]},root,['a', 'b', 'c']\n"
        f"{node_ids[1]},node1,['d', 'e']\n"
        f"{node_ids[2]},node2,None\n"
    )

    assert stream.read() == result
