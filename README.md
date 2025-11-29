## Graph-Based Email Spam Detection

This project implements an email spam detection system using **graph-based features**. The system models emails as nodes in a graph, where edges represent similarity relationships between emails based on their content. Graph-theoretic features are then extracted and used to train a machine learning classifier that distinguishes between spam and non-spam (ham) emails.

### How It Works

1. **Data Preprocessing**: Emails are loaded from a CSV file, text is cleaned (lowercased, URLs removed, punctuation stripped), and subject and body are combined.
2. **Graph Construction**: An optimized graph is built where:
   - Each email is a node
   - Edges connect emails with similar content using Jaccard similarity
   - Only top-K most similar emails are connected per node (optimization)
   - Similarity threshold of 0.20 is used to filter edges
3. **Feature Extraction**: Seven graph-based features are computed for each email:
   - **Degree**: Number of connected emails
   - **Weighted Degree**: Sum of edge weights (similarity scores)
   - **Clustering Coefficient**: Local clustering measure
   - **Betweenness Centrality**: Measure of node importance in the graph
   - **PageRank**: Global importance score
   - **Same Label Neighbors**: Count of neighbors with the same label (spam/ham)
   - **Average Edge Weight**: Mean similarity to connected emails
4. **Classification**: A Logistic Regression classifier is trained on the extracted features to predict spam vs. ham emails.

### Project Structure

- **`main.py`**: Main entry point that orchestrates the entire pipeline:
  - Loads and preprocesses email data
  - Builds the optimized email graph with similarity threshold 0.20 and top_k=5
  - Computes graph features for each email
  - Trains and evaluates the Logistic Regression classifier
  - Saves the trained model and feature columns to disk

- **`Data_Loader.py`**: Handles data loading and preprocessing:
  - `load_data()`: Loads email data from CSV files
  - `clean_text()`: Cleans text by converting to lowercase, removing URLs, punctuation, and extra whitespace
  - `preprocess()`: Combines subject and body columns, normalizes labels, and removes duplicates
  - `load_and_preprocess()`: Convenience function that combines loading and preprocessing

- **`Graph_Builder.py`**: Constructs the email similarity graph:
  - `clean_and_tokenize()`: Tokenizes and filters text (removes stopwords, short words)
  - `jaccard_similarity()`: Computes Jaccard similarity between email word sets
  - `build_email_graph_optimized()`: Builds NetworkX graph with optimized edge creation (top-K similar emails per node)

- **`Graph_Features.py`**: Extracts graph-theoretic features:
  - `compute_graph_features()`: Computes 7 node-level features for each email in the graph
  - Uses NetworkX for centrality measures (betweenness, PageRank, clustering)

- **`Classifier.py`**: Implements the spam detection model:
  - `run_classifier()`: Trains Logistic Regression with 80/20 train/test split
  - Evaluates performance using accuracy, classification report, and confusion matrix
  - Returns trained model and feature column names

### Prerequisites

- Python 3.9+ (recommended)
- A virtual environment tool such as `venv` or `conda`

### Required Dependencies

The project requires the following Python packages:

- `pandas` - Data manipulation and CSV handling
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning (Logistic Regression, train/test split, metrics)
- `networkx` - Graph construction and analysis
- `nltk` - Natural language processing (stopwords)
- `joblib` - Model serialization

If you have a `requirements.txt` file, you can install dependencies with:

```bash
pip install -r requirements.txt
```

Otherwise, install packages individually:

```bash
pip install pandas numpy scikit-learn networkx nltk joblib
```

**Note**: NLTK will automatically download the stopwords corpus on first run.

### Getting Started

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>.git
   cd Graph-Based-Email-Spam-Detection
   ```

2. **(Optional) Create and activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   If you have a `requirements.txt`, run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your dataset**

   - Your dataset should be a CSV file with the following columns:
     - `subject`: Email subject line
     - `body`: Email body content
     - `label`: Spam label (1 or 'spam' for spam, 0 or 'ham' for non-spam)
   - Update the file path in `main.py` (line 15) to point to your dataset:
     ```python
     df_raw = load_and_preprocess("path/to/your/EmailSpamDataset.csv")
     ```

5. **Run the pipeline**

   In the project root, run:

   ```bash
   python main.py
   ```

   The pipeline will:
   - Load and preprocess email data from the CSV file
   - Build an optimized graph connecting similar emails (similarity threshold: 0.20, top-K: 5)
   - Extract 7 graph-based features for each email
   - Save features to `Processed Data.csv`
   - Train a Logistic Regression classifier with 80/20 train/test split
   - Display accuracy, classification report, and confusion matrix
   - Save the trained model to `spam_model.pkl` and feature columns to `feature_columns.pkl`

### Configuration

Key parameters that can be adjusted in the code:

**Graph Construction** (`main.py`, line 18):
- `similarity_threshold`: Minimum Jaccard similarity to create an edge (default: 0.20)
- `top_k`: Maximum number of similar emails to connect per node (default: 5)

**Classifier** (`Classifier.py`, line 12-13):
- `test_size`: Train/test split ratio (default: 0.2, i.e., 80/20 split)
- `random_state`: Random seed for reproducibility (default: 42)
- `max_iter`: Maximum iterations for Logistic Regression (default: 1000)

**Data Path** (`main.py`, line 15):
- Update the CSV file path to your dataset location

**Feature Computation** (`Graph_Features.py`, line 20):
- `k`: Sample size for approximate betweenness centrality (default: 100)

You can modify these parameters directly in the code or extend the project to accept command-line arguments or configuration files.

### Output Files

After running the pipeline, the following files will be generated:

- **`Processed Data.csv`**: Contains the extracted graph features for all emails
- **`spam_model.pkl`**: Serialized trained Logistic Regression model
- **`feature_columns.pkl`**: List of feature column names used by the model

### Using the Trained Model

To use the saved model for prediction on new emails:

```python
import joblib
import pandas as pd
from Data_Loader import load_and_preprocess
from Graph_Builder import build_email_graph_optimized
from Graph_Features import compute_graph_features

# Load model and feature columns
model = joblib.load("spam_model.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# Load and preprocess new data
df_new = load_and_preprocess("new_emails.csv")

# Build graph (you may need to merge with existing graph or build separately)
graph = build_email_graph_optimized(df_new, similarity_threshold=0.20, top_k=5)

# Compute features
labels = {idx: row['label'] for idx, row in df_new.iterrows()}
df_features = compute_graph_features(graph, labels)

# Predict
X = df_features[feature_cols]
predictions = model.predict(X)
```

### Extending the Project

- **New graph features**: Add additional metrics in `Graph_Features.py` (e.g., eigenvector centrality, community detection features, shortest path statistics)
- **Alternative models**: Experiment with different classifiers in `Classifier.py` (e.g., Random Forest, Gradient Boosting, SVM, Neural Networks)
- **Graph optimization**: Implement more efficient graph construction algorithms for larger datasets
- **Visualization**: Add functions to visualize the email graph, feature distributions, and decision boundaries
- **Hyperparameter tuning**: Add grid search or random search for optimal model parameters
- **Cross-validation**: Implement k-fold cross-validation for more robust evaluation
- **Deployment**: Package the trained model behind a REST API or CLI tool for classifying new incoming emails
- **Real-time classification**: Build a system that can classify emails as they arrive without rebuilding the entire graph

### Contributing

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear messages.
4. Open a pull request describing your changes and motivation.

### License

Specify your preferred license here (for example, MIT, Apache 2.0, or proprietary).  
If you are unsure, you can start with an MIT license and adjust later.


