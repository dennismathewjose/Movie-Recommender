import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re

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
        
        # Create a combined search index for keyword-based search
        self.movies_df['search_index'] = self.movies_df.apply(
            lambda x: f"{x['title']} {x['overview']} {x['genres']}",
            axis=1
        )
        
        # Filter out movies with zero ratings
        self.movies_df = self.movies_df[self.movies_df['vote_average'] > 0]
        
        # Ensure plot_embeddings matches movies_df length
        if len(self.plot_embeddings) != len(self.movies_df):
            print("Warning: Mismatch between embeddings and movies_df length")
            # Recalculate embeddings for all movies
            self.plot_embeddings = self.model.encode(self.movies_df['overview'].tolist())
            # Save updated embeddings
            with open(data_path, 'wb') as f:
                data['plot_embeddings'] = self.plot_embeddings
                pickle.dump(data, f)

    def search_by_title(self, query, n=10):
        """
        Search for movies by title
        Args:
            query (str): Movie title to search for
            n (int): Number of results to return
        Returns:
            pd.DataFrame: Matching movies
        """
        try:
            # Simple case-insensitive partial matching
            matches = self.movies_df[
                self.movies_df['title'].str.lower().str.contains(
                    query.lower(),
                    na=False
                )
            ]
            return matches.head(n)
        except Exception as e:
            print(f"Error in search_by_title: {str(e)}")
            return pd.DataFrame()

    def get_similar_movies(self, query, n=10, method='plot', use_ratings=True):
        """
        Get similar movies based on plot or theme
        Args:
            query (str): Search query
            n (int): Number of recommendations
            method (str): 'plot' or 'hybrid'
            use_ratings (bool): Whether to consider ratings in ranking
        Returns:
            pd.DataFrame: Similar movies with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])
            
            # Ensure embeddings are in the correct format
            if len(self.plot_embeddings.shape) == 1:
                self.plot_embeddings = self.plot_embeddings.reshape(1, -1)
            
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
            
            # Sort by similarity score first
            results = results.sort_values('similarity_score', ascending=False)
            
            # If using ratings, sort by rating within similar movies
            if use_ratings and 'vote_average' in results.columns:
                # Create a combined score (70% similarity, 30% rating)
                results['rating_score'] = results['vote_average'] / 10.0
                results['combined_score'] = (
                    0.7 * results['similarity_score'] + 
                    0.3 * results['rating_score']
                )
                results = results.sort_values('combined_score', ascending=False)
            
            # Add genre information
            results['genres'] = results['genres'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else ''
            )
            
            return results[['title', 'genres', 'overview', 'similarity_score', 'vote_average']]
        except Exception as e:
            print(f"Error in get_similar_movies: {str(e)}")
            return pd.DataFrame()

    def search_by_director(self, director_name, n=10):
        """
        Search for movies by director name
        Args:
            director_name (str): Name of the director
            n (int): Number of results to return
        Returns:
            pd.DataFrame: Movies directed by the specified director
        """
        try:
            # Case-insensitive search for director
            matches = self.movies_df[
                self.movies_df['director'].str.lower().str.contains(
                    director_name.lower(),
                    na=False
                )
            ]
            
            if not matches.empty:
                # Sort by rating and return top N
                return matches.sort_values('vote_average', ascending=False).head(n)
            return pd.DataFrame()
        except Exception as e:
            print(f"Error in search_by_director: {str(e)}")
            return pd.DataFrame()

    def search_by_genre(self, genre, n=10):
        """
        Search for movies by genre
        Args:
            genre (str): Genre to search for
            n (int): Number of results to return
        Returns:
            pd.DataFrame: Movies in the specified genre
        """
        try:
            # Convert genre to lowercase for case-insensitive matching
            genre = genre.lower()
            
            # Search for genre in the genres list
            matches = self.movies_df[
                self.movies_df['genres'].apply(
                    lambda x: genre in [g.lower() for g in x] if isinstance(x, list) else False
                )
            ]
            
            if not matches.empty:
                # Sort by rating and return top N
                return matches.sort_values('vote_average', ascending=False).head(n)
            return pd.DataFrame()
        except Exception as e:
            print(f"Error in search_by_genre: {str(e)}")
            return pd.DataFrame()

    def recommend_movies(self, input_text, n=10, method='plot', use_ratings=True):
        """
        Main recommendation function
        Args:
            input_text (str): User input (title, plot, director, or genre)
            n (int): Number of recommendations
            method (str): Recommendation method
            use_ratings (bool): Whether to consider ratings in ranking
        Returns:
            tuple: (exact_matches, similar_movies)
        """
        try:
            # First try exact title matches
            exact_matches = self.search_by_title(input_text, n=n)
            
            if not exact_matches.empty:
                # If exact match found, get similar movies based on its plot
                similar_movies = self.get_similar_movies(
                    exact_matches.iloc[0]['overview'],
                    n=n,
                    method=method,
                    use_ratings=use_ratings
                )
                # Remove the exact match from similar movies if it exists
                if not similar_movies.empty:
                    similar_movies = similar_movies[similar_movies['title'] != exact_matches.iloc[0]['title']]
                return exact_matches, similar_movies
            
            # Try genre search
            genre_matches = self.search_by_genre(input_text, n=n)
            if not genre_matches.empty:
                return pd.DataFrame(), genre_matches
            
            # Try director search
            director_matches = self.search_by_director(input_text, n=n)
            if not director_matches.empty:
                return pd.DataFrame(), director_matches
            
            # If no matches found, get similar movies based on input text
            similar_movies = self.get_similar_movies(
                input_text,
                n=n,
                method=method,
                use_ratings=use_ratings
            )
            
            return pd.DataFrame(), similar_movies
            
        except Exception as e:
            print(f"Error in recommend_movies: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

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
            output.append(f"Rating: {movie['vote_average']:.1f}/10")
            output.append(f"Overview: {movie['overview']}\n")
    
    if not similar_movies.empty:
        output.append("\n=== Similar Movies ===")
        for _, movie in similar_movies.iterrows():
            output.append(f"\nTitle: {movie['title']}")
            output.append(f"Genres: {movie['genres']}")
            output.append(f"Rating: {movie['vote_average']:.1f}/10")
            if 'similarity_score' in movie:
                output.append(f"Similarity Score: {movie['similarity_score']:.2f}")
            output.append(f"Overview: {movie['overview']}\n")
    
    return "\n".join(output)

if __name__ == "__main__":
    # Example usage
    recommender = MovieRecommender('../data/processed_data.pkl')
    
    # Example plot-based search
    print("=== Plot-based Search ===")
    exact_matches, similar_movies = recommender.recommend_movies(
        "A movie about time travel and saving the world",
        use_ratings=True
    )
    print(format_recommendations(exact_matches, similar_movies))
    
    # Example director search
    print("\n=== Director Search ===")
    _, director_movies = recommender.recommend_movies("Christopher Nolan")
    print(format_recommendations(pd.DataFrame(), director_movies))
    
    # Example genre search
    print("\n=== Genre Search ===")
    _, genre_movies = recommender.recommend_movies("Sci-Fi")
    print(format_recommendations(pd.DataFrame(), genre_movies)) 