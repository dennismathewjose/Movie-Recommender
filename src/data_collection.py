import os
import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

class MovieDataCollector:
    def __init__(self):
        self.api_key = os.getenv('TMDB_API_KEY')
        self.base_url = 'https://api.themoviedb.org/3'
        if not self.api_key:
            raise ValueError("TMDB API key not found in environment variables")

    def get_popular_movies(self, num_pages=50):
        """
        Collect popular movies from TMDB API
        Args:
            num_pages (int): Number of pages to collect (20 movies per page)
        Returns:
            list: List of movie dictionaries
        """
        movies = []
        for page in tqdm(range(1, num_pages + 1), desc="Collecting movies"):
            url = f"{self.base_url}/movie/popular"
            params = {
                'api_key': self.api_key,
                'page': page
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                movies.extend(response.json()['results'])
            else:
                print(f"Error fetching page {page}: {response.status_code}")
        return movies

    def get_movie_details(self, movie_id):
        """
        Get detailed information for a specific movie
        Args:
            movie_id (int): TMDB movie ID
        Returns:
            dict: Movie details including plot, genres, etc.
        """
        url = f"{self.base_url}/movie/{movie_id}"
        params = {
            'api_key': self.api_key,
            'append_to_response': 'credits,keywords'
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        return None

    def collect_and_save_data(self, output_path='../data/movies.csv'):
        """
        Collect movie data and save to CSV
        Args:
            output_path (str): Path to save the CSV file
        """
        # Collect basic movie information
        movies = self.get_popular_movies()
        
        # Get detailed information for each movie
        detailed_movies = []
        for movie in tqdm(movies, desc="Collecting movie details"):
            details = self.get_movie_details(movie['id'])
            if details:
                movie_data = {
                    'id': details['id'],
                    'title': details['title'],
                    'overview': details['overview'],
                    'genres': [genre['name'] for genre in details['genres']],
                    'release_date': details['release_date'],
                    'vote_average': details['vote_average'],
                    'runtime': details['runtime']
                }
                detailed_movies.append(movie_data)

        # Convert to DataFrame and save
        df = pd.DataFrame(detailed_movies)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")

if __name__ == "__main__":
    collector = MovieDataCollector()
    collector.collect_and_save_data() 