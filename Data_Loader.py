import pandas as pd
import re

def load_data(path: str):
    """Load CSV dataset"""
    df = pd.read_csv(path, encoding='utf-8')
    return df

'''
This function is used to convert to lowercase , remove urls , punctutations
and extra white-Spaces
'''
def clean_text(text):
   
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)         
    text = re.sub(r"[^a-z0-9\s]", " ", text)    
    text = re.sub(r"\s+", " ", text).strip()  
    return text

'''
This function is used to Combine subject and body, clean text, normalize labels
'''
def preprocess(df, text_cols=['subject','body'], label_col='label'):
    df = df.copy()
    df['text'] = df[text_cols].fillna('').agg(' '.join, axis=1)
    df['text'] = df['text'].apply(clean_text)

    df[label_col] = df[label_col].fillna(0).apply(lambda x: 1 if str(x) in ['1','spam','yes','true','t'] else 0)
    
    df = df[['text', label_col]]
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def load_and_preprocess(path):
    df = load_data(path)
    df = preprocess(df)
    return df
