from typing import List

from probcirc.region_graph.region_graph import RegionGraph
from probcirc.region_graph.rg_node import PartitionNode, RegionNode
from probcirc.region_graph.utils import HypercubeScopeCache

# TODO: add routine for add regions->part->reg structure
# TODO: rework docstrings


def _merge_2_regions(regions: List[RegionNode], graph: RegionGraph) -> RegionNode:
    """Make the structure to connect 2 children.

    Args:
        regions (List[RegionNode]): The children regions.
        graph (nx.DiGraph): The region graph.

    Returns:
        RegionNode: The merged region node.
    """
    assert len(regions) == 2

    scope = regions[0].scope.union(regions[1].scope)
    partition_node = PartitionNode(scope)
    region_node = RegionNode(scope)

    graph.add_edge(regions[0], partition_node)
    graph.add_edge(regions[1], partition_node)
    graph.add_edge(partition_node, region_node)

    return region_node


def _merge_4_regions(regions: List[RegionNode], graph: RegionGraph) -> RegionNode:
    """Make the structure to connect 4 children with structured-decomposability \
        (horizontal then vertical).

    Args:
        regions (List[RegionNode]): The children regions.
        graph (nx.DiGraph): The region graph.

    Returns:
        RegionNode: The merged region node.
    """
    assert len(regions) == 4
    # MERGE regions
    whole_scope = regions[0].scope.union(regions[1].scope).union(regions[2].scope).union(regions[3].scope)
    whole_partition = PartitionNode(whole_scope)
    graph.add_edge(regions[0], whole_partition)
    graph.add_edge(regions[1], whole_partition)
    graph.add_edge(regions[2], whole_partition)
    graph.add_edge(regions[3], whole_partition)

    whole_region = RegionNode(whole_scope)
    graph.add_edge(whole_partition, whole_region)

    return whole_region


def _square_from_buffer(buffer: List[List[RegionNode]], i: int, j: int) -> List[RegionNode]:
    """Get the children of the current position from the buffer.

    Args:
        buffer (List[List[RegionNode]]): The buffer of all children.
        i (int): The i coordinate currently.
        j (int): The j coordinate currently.

    Returns:
        List[RegionNode]: The children nodes.
    """
    children = [buffer[i][j]]
    # TODO: rewrite: len only useful at 2n-1 boundary
    if len(buffer) > i + 1:
        children.append(buffer[i + 1][j])
    if len(buffer[i]) > j + 1:
        children.append(buffer[i][j + 1])
    if len(buffer) > i + 1 and len(buffer[i]) > j + 1:
        children.append(buffer[i + 1][j + 1])
    return children


# pylint: disable-next=too-many-locals,invalid-name
def RealQuadTree(width: int, height: int, final_sum=False) -> RegionGraph:
    """Get quad RG.

        Args:
            width (int): Width of scope.
            height (int): Height of scope.
            struct_decomp (bool, optional): Whether structured-decomposability \
                is required. Defaults to False.

    Returns:
        RegionGraph: The RG.
    """
    assert width == height and width > 0  # TODO: then we don't need two

    shape = (width, height)

    hypercube_to_scope = HypercubeScopeCache()

    buffer: List[List[RegionNode]] = [[] for _ in range(width)]

    graph = RegionGraph()

    # Add Leaves
    for i in range(width):
        for j in range(height):
            hypercube = ((i, j), (i + 1, j + 1))

            c_scope = hypercube_to_scope(hypercube, shape)
            c_node = RegionNode(c_scope)
            graph.add_node(c_node)
            buffer[i].append(c_node)

    lr_choice = 0  # left right # TODO: or choose from 0 and 1?
    td_choice = 0  # top down

    old_buffer_height = height
    old_buffer_width = width
    old_buffer = buffer

    # TODO: also no need to have two for h/w
    while old_buffer_width > 1 and old_buffer_height > 1:  # pylint: disable=while-used
        buffer_height = (old_buffer_height + 1) // 2
        buffer_width = (old_buffer_width + 1) // 2

        buffer = [[] for _ in range(buffer_width)]

        for i in range(buffer_width):
            for j in range(buffer_height):
                regions = _square_from_buffer(old_buffer, 2 * i + lr_choice, 2 * j + td_choice)
                if len(regions) == 1:
                    buf = regions[0]
                elif len(regions) == 2:
                    buf = _merge_2_regions(regions, graph)
                else:
                    buf = _merge_4_regions(regions, graph)
                buffer[i].append(buf)

        old_buffer = buffer
        old_buffer_height = buffer_height
        old_buffer_width = buffer_width

    # add root
    if final_sum:
        roots = list(graph.output_nodes)
        assert len(roots) == 1
        root = roots[0]
        partition_node = PartitionNode(root.scope)
        mixed_root = RegionNode(root.scope)
        graph.add_node(root)
        graph.add_node(partition_node)
        graph.add_edge(root, partition_node)
        graph.add_edge(partition_node, mixed_root)



    assert graph.is_smooth
    assert graph.is_decomposable

    # note: why if adding a final sum is not structured decomposable anymore?
    if not final_sum:
        assert graph.is_structured_decomposable

    return graph
