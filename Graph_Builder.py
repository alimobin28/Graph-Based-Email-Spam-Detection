import networkx as nx
import nltk
import re
from nltk.corpus import stopwords
from collections import defaultdict

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def clean_and_tokenize(text: str):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return set(words)


'''
JS = all common words/all similar words

'''
def jaccard_similarity(set1, set2):
  
    if not set1 or not set2:
        return 0
    return len(set1 & set2) / len(set1 | set2)

    """
    Build optimized email graph:
    1. Apply preprocessing
    2. Compute similarities using Jaccard
    3. Add only top-K similar edges per node
    """
def build_email_graph_optimized(df, similarity_threshold=0.20, top_k=5):
    G = nx.Graph()

    processed = {}
    for idx, row in df.iterrows():
        G.add_node(idx, label=row["label"])
        processed[idx] = clean_and_tokenize(row["text"])

    email_ids = list(processed.keys())

    for i in range(len(email_ids)):
        
        if i % 500 == 0:
            print(f"Processing email {i}/{len(df)}")
            
        e1 = email_ids[i]
        set1 = processed[e1]
    
        similarities = [] 
        for j in range(i + 1, len(email_ids)):
            e2 = email_ids[j]
            sim = jaccard_similarity(set1, processed[e2])

            if sim >= similarity_threshold:
                similarities.append((e2, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_sim = similarities[:top_k]

        for e2, sim in top_sim:
            G.add_edge(e1, e2, weight=sim)

    return G
