import os
import argparse
import json
from data_collection import collect_movie_data
from preprocessing import preprocess_data
from recommender import MovieRecommender

def setup_system(fetch_new_data=False):
    """Set up the recommendation system (one-time setup)."""
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        if fetch_new_data:
            # Collect new data from TMDB
            print("\nCollecting new movie data from TMDB...")
            movies_data = collect_movie_data()
            
            # Save raw data
            print("\nSaving raw data...")
            with open('data/movies.json', 'w', encoding='utf-8') as f:
                json.dump(movies_data, f, indent=2, ensure_ascii=False)
        else:
            # Check if movies.json exists
            if not os.path.exists('data/movies.json'):
                print("Error: movies.json not found in data directory!")
                print("Please ensure you have the movies.json file in the data directory.")
                return
        
        # Preprocess data
        print("\nPreprocessing data...")
        preprocess_data()
        
        print("\nSetup completed successfully!")
        
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        raise

def run_recommendation_system():
    """Run the recommendation system in interactive mode."""
    try:
        # Check if processed data exists
        if not os.path.exists('data/processed_data.pkl'):
            print("\nError: Processed data not found!")
            print("Please run 'python src/main.py --setup' first to set up the system.")
            return
            
        recommender = MovieRecommender()
        
        print("\nWelcome to the Movie Recommendation System!")
        print("==========================================")
        print("You can search for movies in different ways:")
        print("1. Plot-based search: 'A movie about time travel'")
        print("2. Title search: 'The Matrix'")
        print("3. Director search: 'directed by Christopher Nolan'")
        print("4. Genre search: 'action'")
        print("\nType 'quit' to exit")
        print("Type 'help' to see search options again")
        
        while True:
            print("\n" + "="*50)
            query = input("\nEnter your search query: ").strip()
            
            if query.lower() == 'quit':
                print("\nThank you for using the Movie Recommendation System!")
                break
            
            if query.lower() == 'help':
                print("\nSearch Options:")
                print("1. Plot-based search: 'A movie about time travel'")
                print("2. Title search: 'The Matrix'")
                print("3. Director search: 'directed by Christopher Nolan'")
                print("4. Genre search: 'action'")
                continue
            
            if not query:
                print("Please enter a valid query")
                continue
            
            try:
                results = recommender.recommend_movies(query)
                print(recommender.format_recommendations(results))
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error running recommendation system: {str(e)}")
        raise

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument('--setup', action='store_true', help='Set up the system (only needed once)')
    parser.add_argument('--fetch', action='store_true', help='Fetch new data from TMDB API during setup')
    parser.add_argument('--query', type=str, help='Query for movie recommendations')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_system(fetch_new_data=args.fetch)
    elif args.query:
        # Check if processed data exists
        if not os.path.exists('data/processed_data.pkl'):
            print("\nError: Processed data not found!")
            print("Please run 'python src/main.py --setup' first to set up the system.")
            return
            
        try:
            recommender = MovieRecommender()
            results = recommender.recommend_movies(args.query)
            print(recommender.format_recommendations(results))
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
    else:
        # Run in interactive mode by default
        run_recommendation_system()

if __name__ == "__main__":
    main() 