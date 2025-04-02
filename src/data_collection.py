import os
import requests
import pandas as pd
import time
from typing import List, Dict, Optional
from dotenv import load_dotenv

class MovieDataCollector:
    def __init__(self):
        """Initialize the movie data collector with API key."""
        load_dotenv()
        self.api_key = os.getenv('TMDB_API_KEY')
        if not self.api_key:
            raise ValueError("TMDB API key not found in .env file")
        
        self.base_url = "https://api.themoviedb.org/3"
        self.params = {
            'api_key': self.api_key,
            'language': 'en-US'
        }
        
        # Create data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Initialize movies dictionary to store collected movies
        self.movies = {}
        
        # Define major genres
        self.genres = [
            'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
            'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
            'TV Movie', 'Thriller', 'War', 'Western'
        ]

    def fetch_popular_movies(self, pages=50):
        """
        Fetch popular movies from multiple pages.
        
        Args:
            pages (int): Number of pages to fetch (20 movies per page)
        """
        for page in range(1, pages + 1):
            try:
                response = requests.get(
                    f"{self.base_url}/movie/popular",
                    params={
                        'api_key': self.api_key,
                        'page': page,
                        'language': 'en-US'
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                for movie in data['results']:
                    if movie['id'] not in self.movies:
                        self.movies[movie['id']] = self.get_movie_details(movie['id'])
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"Error fetching popular movies page {page}: {str(e)}")
                continue

    def fetch_top_rated_movies(self, pages=50):
        """
        Fetch top-rated movies from multiple pages.
        
        Args:
            pages (int): Number of pages to fetch (20 movies per page)
        """
        for page in range(1, pages + 1):
            try:
                response = requests.get(
                    f"{self.base_url}/movie/top_rated",
                    params={
                        'api_key': self.api_key,
                        'page': page,
                        'language': 'en-US'
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                for movie in data['results']:
                    if movie['id'] not in self.movies:
                        self.movies[movie['id']] = self.get_movie_details(movie['id'])
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"Error fetching top-rated movies page {page}: {str(e)}")
                continue

    def fetch_movies_by_genres(self, pages_per_genre=10):
        """
        Fetch movies from various genres.
        
        Args:
            pages_per_genre (int): Number of pages to fetch per genre
        """
        genres = [
            'Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
            'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
            'TV Movie', 'Thriller', 'War', 'Western'
        ]
        
        for genre in genres:
            for page in range(1, pages_per_genre + 1):
                try:
                    response = requests.get(
                        f"{self.base_url}/discover/movie",
                        params={
                            'api_key': self.api_key,
                            'with_genres': self._get_genre_id(genre),
                            'page': page,
                            'language': 'en-US',
                            'sort_by': 'vote_average.desc'  # Get highest rated movies first
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    for movie in data['results']:
                        if movie['id'] not in self.movies:
                            self.movies[movie['id']] = self.get_movie_details(movie['id'])
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    print(f"Error fetching {genre} movies page {page}: {str(e)}")
                    continue

    def collect_movie_data(self):
        """Collect movie data from various sources."""
        try:
            # Initialize counters
            total_attempted = 0
            total_successful = 0
            failed_movies = []
            
            print("\nStarting movie collection process...")
            
            # Fetch popular movies (50 pages = 1000 movies)
            print("1. Fetching popular movies...")
            popular_count = len(self.movies)
            self.fetch_popular_movies(pages=50)
            total_attempted += 1000
            total_successful = len(self.movies)
            print(f"Popular movies collected: {total_successful - popular_count}")
            
            # Fetch top-rated movies (50 pages = 1000 movies)
            print("2. Fetching top-rated movies...")
            top_rated_count = len(self.movies)
            self.fetch_top_rated_movies(pages=50)
            total_attempted += 1000
            total_successful = len(self.movies)
            print(f"Top-rated movies collected: {total_successful - top_rated_count}")
            
            # Fetch genre-based movies (19 genres × 10 pages = 1900 movies)
            print("3. Fetching genre-based movies...")
            genre_count = len(self.movies)
            self.fetch_movies_by_genres(pages_per_genre=10)
            total_attempted += 1900
            total_successful = len(self.movies)
            print(f"Genre-based movies collected: {total_successful - genre_count}")
            
            # Convert to list of dictionaries
            print("\nConverting collected data...")
            movies_data = []
            for movie_id, movie in self.movies.items():
                if movie is not None:  # Skip None values
                    try:
                        movies_data.append({
                            'id': movie_id,
                            'title': movie['title'],
                            'overview': movie['overview'],
                            'genres': [genre['name'] for genre in movie['genres']],
                            'director': movie.get('director', 'Unknown'),
                            'vote_average': movie['vote_average'],
                            'release_date': movie['release_date'],
                            'cast': movie.get('cast', []),
                            'keywords': movie.get('keywords', []),
                            'poster_path': movie.get('poster_path', ''),
                            'backdrop_path': movie.get('backdrop_path', ''),
                            'runtime': movie.get('runtime', 0),
                            'budget': movie.get('budget', 0),
                            'revenue': movie.get('revenue', 0),
                            'production_companies': [company['name'] for company in movie.get('production_companies', [])],
                            'languages': movie.get('spoken_languages', [])
                        })
                    except Exception as e:
                        print(f"Error processing movie {movie_id}: {str(e)}")
                        failed_movies.append(movie_id)
            
            # Remove duplicates based on ID
            unique_movies = {movie['id']: movie for movie in movies_data}.values()
            movies_data = list(unique_movies)
            duplicates_removed = len(movies_data) - len(unique_movies)
            
            print(f"\nCollection Summary:")
            print(f"Total movies attempted: {total_attempted}")
            print(f"Total movies successfully collected: {total_successful}")
            print(f"Failed to collect: {len(failed_movies)}")
            print(f"Duplicates removed: {duplicates_removed}")
            print(f"Final unique movies: {len(movies_data)}")
            
            if failed_movies:
                print("\nFailed movie IDs:")
                print(failed_movies[:10])  # Show first 10 failed IDs
            
            print("\nGenre distribution:")
            all_genres = [genre for movie in movies_data for genre in movie['genres']]
            genre_counts = pd.Series(all_genres).value_counts()
            print(genre_counts)
            
            return movies_data
            
        except Exception as e:
            print(f"Error collecting movie data: {str(e)}")
            raise

    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """
        Get detailed information about a movie.
        
        Args:
            movie_id (int): TMDB movie ID
            
        Returns:
            Optional[Dict]: Movie details dictionary
        """
        try:
            # Fetch movie details
            response = requests.get(
                f"{self.base_url}/movie/{movie_id}",
                params={**self.params, 'append_to_response': 'credits,keywords'}
            )
            response.raise_for_status()
            movie_data = response.json()
            
            # Skip movies without overview
            if not movie_data.get('overview'):
                print(f"Skipping movie {movie_id}: No overview available")
                return None
            
            # Extract relevant information
            movie_details = {
                'id': movie_data['id'],
                'title': movie_data['title'],
                'overview': movie_data['overview'],
                'release_date': movie_data['release_date'],
                'vote_average': movie_data['vote_average'],
                'runtime': movie_data.get('runtime', 0),
                'genres': movie_data['genres'],
                'director': self._get_director(movie_data.get('credits', {})),
                'cast': self._get_cast(movie_data.get('credits', {})),
                'keywords': [keyword['name'] for keyword in movie_data.get('keywords', {}).get('keywords', [])]
            }
            
            return movie_details
            
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching movie {movie_id}: {str(e)}")
            return None
        except Exception as e:
            print(f"Error fetching details for movie {movie_id}: {str(e)}")
            return None

    def _get_director(self, credits: Dict) -> str:
        """Extract director name from credits."""
        try:
            for crew in credits.get('crew', []):
                if crew['job'] == 'Director':
                    return crew['name']
            return "Unknown"
        except:
            return "Unknown"

    def _get_cast(self, credits: Dict) -> List[str]:
        """Extract top cast members from credits."""
        try:
            return [cast['name'] for cast in credits.get('cast', [])[:5]]
        except:
            return []

    def save_movies(self, movies: List[Dict], filename: str = 'movies.json'):
        """
        Save movies to JSON file.
        
        Args:
            movies (List[Dict]): List of movie data dictionaries
            filename (str): Output filename
        """
        try:
            import json
            
            # Save to JSON with proper encoding and formatting
            with open(f'data/{filename}', 'w', encoding='utf-8') as f:
                json.dump(movies, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(movies)} movies to data/{filename}")
        except Exception as e:
            print(f"Error saving movies: {str(e)}")
            raise

    def _get_genre_id(self, genre_name: str) -> str:
        """Get TMDB genre ID from genre name."""
        genre_ids = {
            'Action': '28',
            'Adventure': '12',
            'Animation': '16',
            'Comedy': '35',
            'Crime': '80',
            'Documentary': '99',
            'Drama': '18',
            'Family': '10751',
            'Fantasy': '14',
            'History': '36',
            'Horror': '27',
            'Music': '10402',
            'Mystery': '9648',
            'Romance': '10749',
            'Science Fiction': '878',
            'TV Movie': '10770',
            'Thriller': '53',
            'War': '10752',
            'Western': '37'
        }
        return genre_ids.get(genre_name, '')

def collect_movie_data():
    """Main function to collect and save movie data."""
    try:
        collector = MovieDataCollector()
        
        # Collect movie data
        print("Collecting movie data...")
        movies_data = collector.collect_movie_data()
        
        # Save to JSON
        collector.save_movies(movies_data)
        print(f"Successfully collected {len(movies_data)} unique movies")
        
        return movies_data
        
    except Exception as e:
        print(f"Error collecting movie data: {str(e)}")
        raise

if __name__ == "__main__":
    collect_movie_data() 