import os
import requests
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import time

class MovieDataCollector:
    def __init__(self):
        """
        Initialize the movie data collector with TMDB API key
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key
        self.api_key = os.getenv('TMDB_API_KEY')
        if not self.api_key:
            raise ValueError("TMDB_API_KEY not found in environment variables")
        
        # Set up base URL
        self.base_url = "https://api.themoviedb.org/3"
        
        # Set up headers
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def get_popular_movies(self, num_pages=50):
        """
        Get popular movies from TMDB API
        Args:
            num_pages (int): Number of pages to fetch
        Returns:
            list: List of movie dictionaries
        """
        movies = []
        
        # Fetch popular movies
        for page in tqdm(range(1, num_pages + 1), desc="Collecting movies"):
            url = f"{self.base_url}/movie/popular"
            params = {
                'api_key': self.api_key,
                'page': page
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                movies.extend(data['results'])
            else:
                print(f"Error fetching page {page}: {response.status_code}")
            
            # Add delay to respect rate limits
            time.sleep(0.1)
        
        return movies

    def get_movie_details(self, movie_id):
        """
        Get detailed information for a specific movie
        Args:
            movie_id (int): TMDB movie ID
        Returns:
            dict: Detailed movie information
        """
        url = f"{self.base_url}/movie/{movie_id}"
        params = {
            'api_key': self.api_key,
            'append_to_response': 'credits,keywords'
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching movie {movie_id}: {response.status_code}")
            return None

    def collect_and_save_data(self, output_path):
        """
        Collect movie data and save to CSV
        Args:
            output_path (str): Path to save the CSV file
        """
        # Get popular movies
        movies = self.get_popular_movies()
        
        # Collect detailed information
        detailed_movies = []
        for movie in tqdm(movies, desc="Collecting movie details"):
            details = self.get_movie_details(movie['id'])
            
            if details:
                # Get genres
                genres = [genre['name'] for genre in details.get('genres', [])]
                
                # Get director from credits
                director = 'Unknown'
                if 'credits' in details and 'crew' in details['credits']:
                    director = next(
                        (person['name'] for person in details['credits']['crew'] 
                         if person['job'] == 'Director'),
                        'Unknown'
                    )
                
                # Get cast members
                cast = []
                if 'credits' in details and 'cast' in details['credits']:
                    cast = [person['name'] for person in details['credits']['cast'][:5]]
                
                # Create movie data dictionary
                movie_data = {
                    'id': details['id'],
                    'title': details['title'],
                    'overview': details['overview'],
                    'genres': genres,
                    'release_date': details.get('release_date', 'Unknown'),
                    'vote_average': details.get('vote_average', 0.0),
                    'runtime': details.get('runtime', 0),
                    'director': director,
                    'cast': cast
                }
                
                detailed_movies.append(movie_data)
            
            # Add delay to respect rate limits
            time.sleep(0.1)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(detailed_movies)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(detailed_movies)} movies to {output_path}")

if __name__ == "__main__":
    # Example usage
    collector = MovieDataCollector()
    collector.collect_and_save_data('data/movies.csv') 