from Data_Loader import load_and_preprocess
from Graph_Builder import build_email_graph_optimized
from Graph_Features import compute_graph_features
from Classifier import run_classifier  
import joblib
'''
The main functions consists off 4 main steps 
1) load and process data from pandas and return processed data
2) Build optimized Graph with threshold of 50 words
3) Find its features 
4) apply logistic regression
'''
def main():
    
    df_raw = load_and_preprocess(r"C:\Users\alimo\OneDrive\Desktop\EmailSpamDataset.csv")
    print(f"Dataset loaded with {len(df_raw)} emails")
    
    graph = build_email_graph_optimized(df_raw, word_threshold=15)
    print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    labels = {idx: row['label'] for idx, row in df_raw.iterrows()}

    df_features = compute_graph_features(graph, labels)
    print("Graph features computed:")
    print(df_features.head())
    
    df_features.to_csv(r"C:\Users\alimo\OneDrive\Desktop\Processed Data.csv", index=False)
    print("Graph features saved to Processed Data.csv")
    
    model, feature_cols = run_classifier(df_features)
    joblib.dump(model, "spam_model.pkl")
    joblib.dump(feature_cols, "feature_columns.pkl")
    
    print("Model saved.")

if __name__ == "__main__":
    main()
