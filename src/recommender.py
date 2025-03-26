import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import torch

class MovieRecommender:
    def __init__(self, data_path):
        """
        Initialize the movie recommender system
        Args:
            data_path (str): Path to the processed data file
        """
        # Load processed data
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.movies_df = data['movies_df']
        self.genre_features = data['genre_features']
        self.plot_embeddings = data['plot_embeddings']
        self.mlb = data['mlb']
        self.model = data['model']

    def search_by_title(self, query, n=10):
        """
        Search for movies by title
        Args:
            query (str): Movie title to search for
            n (int): Number of results to return
        Returns:
            pd.DataFrame: Matching movies
        """
        # Simple case-insensitive partial matching
        matches = self.movies_df[
            self.movies_df['title'].str.lower().str.contains(
                query.lower(),
                na=False
            )
        ]
        return matches.head(n)

    def get_similar_movies(self, query, n=10, method='plot'):
        """
        Get similar movies based on plot or theme
        Args:
            query (str): Search query
            n (int): Number of recommendations
            method (str): 'plot' or 'hybrid'
        Returns:
            pd.DataFrame: Similar movies with similarity scores
        """
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(
            query_embedding,
            self.plot_embeddings
        )[0]
        
        # Get top similar movies
        similar_indices = np.argsort(similarities)[::-1][:n]
        similar_scores = similarities[similar_indices]
        
        # Create results dataframe
        results = self.movies_df.iloc[similar_indices].copy()
        results['similarity_score'] = similar_scores
        
        # Add genre information
        results['genres'] = results['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
        
        return results[['title', 'genres', 'overview', 'similarity_score']]

    def recommend_movies(self, input_text, n=10, method='plot'):
        """
        Main recommendation function
        Args:
            input_text (str): User input (title, plot, or theme)
            n (int): Number of recommendations
            method (str): Recommendation method
        Returns:
            tuple: (exact_matches, similar_movies)
        """
        # First try exact title matches
        exact_matches = self.search_by_title(input_text, n=n)
        
        # Then find similar movies based on plot/theme
        similar_movies = self.get_similar_movies(input_text, n=n, method=method)
        
        # Remove duplicates from similar_movies if they appear in exact_matches
        if not exact_matches.empty:
            similar_movies = similar_movies[
                ~similar_movies['title'].isin(exact_matches['title'])
            ]
        
        return exact_matches, similar_movies

def format_recommendations(exact_matches, similar_movies):
    """
    Format the recommendations for display
    Args:
        exact_matches (pd.DataFrame): Exact title matches
        similar_movies (pd.DataFrame): Similar movies
    Returns:
        str: Formatted recommendations
    """
    output = []
    
    if not exact_matches.empty:
        output.append("=== Exact Matches ===")
        for _, movie in exact_matches.iterrows():
            output.append(f"\nTitle: {movie['title']}")
            output.append(f"Genres: {movie['genres']}")
            output.append(f"Overview: {movie['overview']}\n")
    
    if not similar_movies.empty:
        output.append("\n=== Similar Movies ===")
        for _, movie in similar_movies.iterrows():
            output.append(f"\nTitle: {movie['title']}")
            output.append(f"Genres: {movie['genres']}")
            output.append(f"Similarity Score: {movie['similarity_score']:.2f}")
            output.append(f"Overview: {movie['overview']}\n")
    
    return "\n".join(output)

if __name__ == "__main__":
    # Example usage
    recommender = MovieRecommender('../data/processed_data.pkl')
    
    # Example search
    query = "A science fiction movie about time travel"
    exact_matches, similar_movies = recommender.recommend_movies(query)
    
    # Print results
    print(format_recommendations(exact_matches, similar_movies)) 