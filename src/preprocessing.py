import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import re
import ast
from sentence_transformers import SentenceTransformer
import torch
import pickle

class MovieDataPreprocessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the preprocessor with a BERT model
        Args:
            model_name (str): Name of the SBERT model to use
        """
        self.model = SentenceTransformer(model_name)
        self.mlb = MultiLabelBinarizer()

    def clean_text(self, text):
        """
        Clean and normalize text data
        Args:
            text (str): Input text
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub('<[^<]+?>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        text = re.sub('\s+', ' ', text)
        
        return text

    def process_genres(self, genres):
        """
        Process genre strings into a list
        Args:
            genres (str): String representation of genres list
        Returns:
            list: List of genres
        """
        if isinstance(genres, str):
            try:
                return ast.literal_eval(genres)
            except:
                return []
        return genres if isinstance(genres, list) else []

    def generate_embeddings(self, texts):
        """
        Generate BERT embeddings for a list of texts
        Args:
            texts (list): List of text strings
        Returns:
            numpy.ndarray: Array of embeddings
        """
        return self.model.encode(texts, show_progress_bar=True)

    def preprocess_data(self, input_path, output_path):
        """
        Preprocess the movie data and save the results
        Args:
            input_path (str): Path to input CSV file
            output_path (str): Path to save processed data
        """
        # Load data
        df = pd.read_csv(input_path)
        
        # Clean plot summaries
        df['cleaned_overview'] = df['overview'].apply(self.clean_text)
        
        # Process genres
        df['genres'] = df['genres'].apply(self.process_genres)
        
        # Generate genre encodings
        genre_encodings = self.mlb.fit_transform(df['genres'])
        genre_features = pd.DataFrame(
            genre_encodings,
            columns=self.mlb.classes_
        )
        
        # Generate plot embeddings
        plot_embeddings = self.generate_embeddings(df['cleaned_overview'].tolist())
        
        # Save processed data
        processed_data = {
            'movies_df': df,
            'genre_features': genre_features,
            'plot_embeddings': plot_embeddings,
            'mlb': self.mlb,
            'model': self.model
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    preprocessor = MovieDataPreprocessor()
    preprocessor.preprocess_data(
        input_path='../data/movies.csv',
        output_path='../data/processed_data.pkl'
    ) 