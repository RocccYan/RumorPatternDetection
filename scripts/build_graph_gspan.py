import os
import json

from utils import read_json, save_as_json, mkdir

"""
this is to build graph file in gspan style for stored edges of social media sessions.
"""


def build_graph_gspan(nodes, edges):
    """
    Build a gspan graph.
    Suppose a graph is introduced by nodes and edges, in which 
        - nodes are stored by (node_id, label) or just node_id.
        - edges are stored like (src, dst, label) or (src, dst).
    For gspan graph, node and graph are stored line by line in a text file, whose demo is shown in gSpan package.
    Return gspan graph and a mapping dictionary of nodes_id.
    """
    default_label = 2

    if len(nodes[0]) == 2:
        nodes_map = {node_id:idx for idx, (node_id,_) in enumerate(nodes)}
        gspan_nodes = "".join(f"v {nodes_map[node[0]]} {node[1]}\n" for node in nodes)
    else:
        nodes_map = {node_id:idx for idx, node_id in enumerate(nodes)}
        gspan_nodes = "".join(f"v {nodes_map[node]} {default_label}\n" for node in nodes)
    
    # case when edges have label.
    if len(edges[0]) == 3:
        gspan_edges = "".join(f"e {nodes_map[src]} {nodes_map[dst]} {label}\n" for (src,dst, label) in edges)
    else:
        gspan_edges = "".join(f"e {nodes_map[src]} {nodes_map[dst]} {default_label}\n" for (src,dst) in edges)

    return (gspan_nodes + gspan_edges, nodes_map)


def main():
    # load raw files of graphs.
    raw_graph_filepath = './data/CED_Dataset/session_structures.json'
    session_structures = read_json(raw_graph_filepath)
    session_labels = read_json('./data/CED_Dataset/session_labels.json')
    rumor_id = [session_id for (session_id, label) in session_labels.items() if label==1]
    non_rumor_id = [session_id for (session_id, label) in session_labels.items() if label==0]
    gspan_graphs = ""

    gspan_graphs_nodes_map = {}
    for graph_idx, (session_id, edges) in enumerate(session_structures.items()):
        if session_id in non_rumor_id:
            # build nodes and edges according to session information.
            nodes = list({node for node,_ in edges} | {node for _,node in edges})
            gspan_graph, nodes_map = build_graph_gspan(nodes, edges)
            gspan_graphs += f"t # {graph_idx}\n{gspan_graph}"
            gspan_graphs_nodes_map[session_id] = nodes_map

    gspan_graphs += "t # -1\n"

    with open('./data/CED_Dataset/graph_gspan_non_rumor.data', 'w', encoding='utf-8') as fh:
        fh.write(gspan_graphs)

        # save_as_json(gspan_graphs_nodes_map,'./data/CED_Dataset/graph_map_gspan.json')


if __name__ == "__main__":
    main()