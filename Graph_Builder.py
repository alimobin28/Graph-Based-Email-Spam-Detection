import networkx as nx
from collections import defaultdict
from Data_Loader import load_and_preprocess

# Define a small set of stopwords to ignore common words
STOPWORDS = set([
    "the", "and", "to", "a", "is", "in", "on", "for", "of", 
    "with", "this", "that", "it", "as", "at", "by", "an"
])

"""
    Optimized email graph builder using inverted index and stopwords removal.

    Args:
        df: preprocessed dataframe with columns 'text' and 'label'
        word_threshold: minimum number of shared words to create an edge
"""
def build_email_graph_optimized(df, word_threshold=5):
    G = nx.Graph()
    
    # Add nodes with label
    for idx, row in df.iterrows():
        G.add_node(idx, label=row['label'])
    
    # Convert text to sets of words and remove stopwords
    texts = df['text'].apply(lambda x: set(word for word in str(x).split() if word not in STOPWORDS))
    
    # Build inverted index: word -> set of email indices containing it
    inverted_index = defaultdict(set)
    for idx, word_set in texts.items():
        for word in word_set:
            inverted_index[word].add(idx)
    
    # Keep track of edges already added
    added_edges = set()
    
    # Build edges based on shared words
    for i in range(len(df)):
        # DEBUG PRINT: show progress every 500 emails
        if i % 500 == 0:
            print(f"Processing email {i}/{len(df)}")
        
        neighbors = defaultdict(int)  # email idx -> count of shared words
        
        for word in texts[i]:
            for j in inverted_index[word]:
                if i != j:
                    neighbors[j] += 1
        
        # Add edges if threshold is met
        for j, count in neighbors.items():
            if count >= word_threshold:
                edge = tuple(sorted((i, j)))
                if edge not in added_edges:
                    G.add_edge(i, j, type='text', weight=count)
                    added_edges.add(edge)
    
    return G
