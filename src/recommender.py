import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

class MovieRecommender:
    def __init__(self):
        """Initialize the movie recommender system."""
        try:
            # Load processed data
            with open('data/processed_data.pkl', 'rb') as f:
                data = pickle.load(f)
            
            self.movies_df = data['movies_df']
            self.genre_features = data['genre_features']
            self.plot_embeddings = data['plot_embeddings']
            self.mlb = data['mlb']
            self.model = data['model']
            
            # Create a combined search index for keyword-based search
            self.movies_df['search_index'] = self.movies_df.apply(
                lambda x: f"{x['title']} {x['overview']} {' '.join(x['genres'])}",
                axis=1
            )
            
            # Filter out movies with zero ratings
            self.movies_df = self.movies_df[self.movies_df['vote_average'] > 0]
            
            # Ensure plot_embeddings matches movies_df length
            if len(self.plot_embeddings) != len(self.movies_df):
                print("Warning: Mismatch between embeddings and movies_df length")
                # Use the smaller length
                min_len = min(len(self.plot_embeddings), len(self.movies_df))
                self.movies_df = self.movies_df.iloc[:min_len].reset_index(drop=True)
                self.plot_embeddings = self.plot_embeddings[:min_len]
            
            print("Recommender initialized successfully!")
        except Exception as e:
            print(f"Error initializing recommender: {str(e)}")
            raise

    def search_by_title(self, query, n=5):
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

    def get_similar_movies(self, query, n=5, method='plot', use_ratings=True):
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
            
            # Format genre list if needed
            if 'genres' in results.columns:
                results['genres_display'] = results['genres'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else ''
                )
            
            return results
        except Exception as e:
            print(f"Error in get_similar_movies: {str(e)}")
            return pd.DataFrame()

    def search_by_director(self, director_name, n=5):
        """
        Search for movies by director name
        Args:
            director_name (str): Name of the director
            n (int): Number of results to return
        Returns:
            pd.DataFrame: Movies directed by the specified director
        """
        try:
            # Extract the director name from the query if necessary
            director_name = director_name.lower()
            for phrase in ['directed by', 'director', 'films by', 'movies by']:
                director_name = director_name.replace(phrase, '').strip()
            
            # Case-insensitive search for director
            matches = self.movies_df[
                self.movies_df['director'].str.lower().str.contains(
                    director_name,
                    na=False
                )
            ]
            
            if not matches.empty:
                # Format genre list
                matches['genres_display'] = matches['genres'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else ''
                )
                # Sort by rating and return top N
                return matches.sort_values('vote_average', ascending=False).head(n)
            
            print(f"No movies found for director: {director_name}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error in search_by_director: {str(e)}")
            return pd.DataFrame()

    def search_by_genre(self, genre, n=5):
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
                    lambda x: any(genre in g.lower() for g in x) if isinstance(x, list) else False
                )
            ]
            
            if not matches.empty:
                # Format genre list
                matches['genres_display'] = matches['genres'].apply(
                    lambda x: ', '.join(x) if isinstance(x, list) else ''
                )
                # Sort by rating and return top N
                return matches.sort_values('vote_average', ascending=False).head(n)
            
            print(f"No movies found for genre: {genre}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error in search_by_genre: {str(e)}")
            return pd.DataFrame()

    def recommend_movies(self, input_text, n=5):
        """
        Main recommendation function
        Args:
            input_text (str): User input (title, plot, director, or genre)
            n (int): Number of recommendations
        Returns:
            tuple: (recommendation_type, movies)
        """
        try:
            # First try exact title matches
            exact_matches = self.search_by_title(input_text, n=1)
            
            if not exact_matches.empty:
                print(f"Found exact title match: {exact_matches.iloc[0]['title']}")
                # If exact match found, get similar movies based on its plot
                similar_movies = self.get_similar_movies(
                    exact_matches.iloc[0]['overview'],
                    n=n+1  # Get extra to filter out the exact match
                )
                # Remove the exact match from similar movies if it exists
                if not similar_movies.empty:
                    similar_movies = similar_movies[similar_movies['title'] != exact_matches.iloc[0]['title']]
                    similar_movies = similar_movies.head(n)
                
                return ('title', exact_matches, similar_movies)
            
            # If no title match, try director search directly
            director_movies = self.search_by_director(input_text, n=n)
            if not director_movies.empty:
                return ('director', pd.DataFrame(), director_movies)
            
            # Try genre search (check if input matches any known genre)
            all_genres = set()
            for genres in self.movies_df['genres']:
                if isinstance(genres, list):
                    all_genres.update([g.lower() for g in genres])
            
            if input_text.lower() in all_genres:
                genre_movies = self.search_by_genre(input_text, n=n)
                if not genre_movies.empty:
                    return ('genre', pd.DataFrame(), genre_movies)
            
            # If no matches found, get similar movies based on input text
            similar_movies = self.get_similar_movies(input_text, n=n)
            
            return ('plot', pd.DataFrame(), similar_movies)
            
        except Exception as e:
            print(f"Error in recommend_movies: {str(e)}")
            return ('error', pd.DataFrame(), pd.DataFrame())
    
    def format_recommendations(self, results):
        """Format the recommendations for display"""
        rec_type, exact_matches, similar_movies = results
        
        output = []
        
        if rec_type == 'title':
            if not exact_matches.empty:
                output.append("=== Exact Match ===")
                movie = exact_matches.iloc[0]
                output.append(f"\nTitle: {movie['title']}")
                
                genres_display = ', '.join(movie['genres']) if isinstance(movie['genres'], list) else movie['genres']
                output.append(f"Genres: {genres_display}")
                
                if 'director' in movie and pd.notna(movie['director']):
                    output.append(f"Director: {movie['director']}")
                
                output.append(f"Rating: {movie['vote_average']:.1f}/10")
                output.append(f"Overview: {movie['overview']}\n")
            
            if not similar_movies.empty:
                output.append("\n=== Similar Movies ===")
                for _, movie in similar_movies.iterrows():
                    output.append(f"\nTitle: {movie['title']}")
                    
                    genres_display = ', '.join(movie['genres']) if isinstance(movie['genres'], list) else movie['genres']
                    output.append(f"Genres: {genres_display}")
                    
                    if 'director' in movie and pd.notna(movie['director']):
                        output.append(f"Director: {movie['director']}")
                    
                    output.append(f"Rating: {movie['vote_average']:.1f}/10")
                    output.append(f"Similarity: {movie['similarity_score']:.2f}")
                    output.append(f"Overview: {movie['overview']}\n")
        
        elif rec_type == 'director':
            if not similar_movies.empty:
                output.append(f"=== Top Movies by {similar_movies.iloc[0]['director']} ===")
                for _, movie in similar_movies.iterrows():
                    output.append(f"\nTitle: {movie['title']}")
                    
                    genres_display = ', '.join(movie['genres']) if isinstance(movie['genres'], list) else movie['genres']
                    output.append(f"Genres: {genres_display}")
                    
                    output.append(f"Rating: {movie['vote_average']:.1f}/10")
                    output.append(f"Overview: {movie['overview']}\n")
        
        elif rec_type == 'genre':
            if not similar_movies.empty:
                genre_name = similar_movies.iloc[0]['genres'][0] if isinstance(similar_movies.iloc[0]['genres'], list) else ""
                output.append(f"=== Top {genre_name} Movies ===")
                for _, movie in similar_movies.iterrows():
                    output.append(f"\nTitle: {movie['title']}")
                    
                    genres_display = ', '.join(movie['genres']) if isinstance(movie['genres'], list) else movie['genres']
                    output.append(f"Genres: {genres_display}")
                    
                    if 'director' in movie and pd.notna(movie['director']):
                        output.append(f"Director: {movie['director']}")
                    
                    output.append(f"Rating: {movie['vote_average']:.1f}/10")
                    output.append(f"Overview: {movie['overview']}\n")
        
        elif rec_type == 'plot':
            if not similar_movies.empty:
                output.append("=== Movies Matching Your Description ===")
                for _, movie in similar_movies.iterrows():
                    output.append(f"\nTitle: {movie['title']}")
                    
                    genres_display = ', '.join(movie['genres']) if isinstance(movie['genres'], list) else movie['genres']
                    output.append(f"Genres: {genres_display}")
                    
                    if 'director' in movie and pd.notna(movie['director']):
                        output.append(f"Director: {movie['director']}")
                    
                    output.append(f"Rating: {movie['vote_average']:.1f}/10")
                    output.append(f"Similarity: {movie['similarity_score']:.2f}")
                    output.append(f"Overview: {movie['overview']}\n")
        
        return "\n".join(output) if output else "No movies found matching your query." 