import pandas as pd
import numpy as np
import re
import pickle
import json
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer

class TextCleaner:
    def __init__(self):
        """Initialize the text cleaner with NLTK stopwords."""
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and standardize text."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(words)

def preprocess_data():
    """Preprocess the collected movie data."""
    try:
        print("\nLoading movie data...")
        with open('data/movies.json', 'r', encoding='utf-8') as f:
            movies_data = json.load(f)
        
        print("Creating DataFrame...")
        # Convert to DataFrame
        movies_df = pd.DataFrame(movies_data)
        
        # Initialize text cleaner
        cleaner = TextCleaner()
        
        # Clean overview text (but keep original for display)
        print("Cleaning text data...")
        movies_df['clean_overview'] = movies_df['overview'].apply(
            lambda x: cleaner.clean_text(x) if isinstance(x, str) else ""
        )
        
        # Initialize SentenceTransformer model
        print("Loading NLP model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate plot embeddings
        print("Generating plot embeddings...")
        plot_embeddings = model.encode(
            movies_df['clean_overview'].tolist(),
            show_progress_bar=True
        )
        
        # Prepare genre features
        print("Processing genres...")
        # Ensure genres is a list
        movies_df['genres'] = movies_df['genres'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        
        # Create genre one-hot encoding
        mlb = MultiLabelBinarizer()
        genre_features = mlb.fit_transform(movies_df['genres'])
        
        # Filter out invalid entries
        print("Filtering data...")
        valid_mask = (
            movies_df['overview'].notna() & 
            movies_df['title'].notna() & 
            (movies_df['overview'].str.len() > 0)
        )
        movies_df = movies_df[valid_mask].reset_index(drop=True)
        plot_embeddings = plot_embeddings[valid_mask]
        genre_features = genre_features[valid_mask]
        
        # Save processed data
        print("\nSaving processed data...")
        processed_data = {
            'movies_df': movies_df,
            'plot_embeddings': plot_embeddings,
            'genre_features': genre_features,
            'mlb': mlb,
            'model': model
        }
        
        with open('data/processed_data.pkl', 'wb') as f:
            pickle.dump(processed_data, f)
        
        print("Data preprocessing completed successfully!")
        
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_data() 