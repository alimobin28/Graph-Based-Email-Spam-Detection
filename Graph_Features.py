import networkx as nx
import pandas as pd
from Graph_Builder import build_email_graph_optimized
from Data_Loader import load_and_preprocess

"""
    Compute 7 node-level features for each email/node in the graph.

    Args:
        G : networkx.Graph - email graph
        labels : dict - node label mapping {node: label}

    Returns:
        df_features : pandas.DataFrame - features for each node
    """
def compute_graph_features(G, labels):
   
    features = []

    betweenness = nx.betweenness_centrality(G, k=100, seed=42)
    pagerank = nx.pagerank(G)

    for idx, node in enumerate(G.nodes()):
     
        if idx % 500 == 0:
            print(f"Computing features for node {idx}/{len(G.nodes())}")

        neighbors = list(G.neighbors(node))
        degree = len(neighbors)
        weighted_degree = sum(G[node][nbr]['weight'] for nbr in neighbors)
        clustering = nx.clustering(G, node)
        bc = betweenness[node]
        pr = pagerank[node]

        same_label_count = sum(1 for nbr in neighbors if labels[nbr] == labels[node])

        avg_edge_weight = weighted_degree / degree if degree > 0 else 0

        features.append({
            'node': node,
            'degree': degree,
            'weighted_degree': weighted_degree,
            'clustering': clustering,
            'betweenness': bc,
            'pagerank': pr,
            'same_label_neighbors': same_label_count,
            'avg_edge_weight': avg_edge_weight,
            'label': labels[node] 
        })

    df_features = pd.DataFrame(features)
    return df_features
