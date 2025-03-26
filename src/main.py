import os
import argparse
from data_collection import MovieDataCollector
from preprocessing import MovieDataPreprocessor
from recommender import MovieRecommender, format_recommendations

def setup_data_collection(args):
    """
    Set up and run data collection
    """
    collector = MovieDataCollector()
    collector.collect_and_save_data(output_path=args.movies_path)

def setup_preprocessing(args):
    """
    Set up and run preprocessing
    """
    preprocessor = MovieDataPreprocessor()
    preprocessor.preprocess_data(
        input_path=args.movies_path,
        output_path=args.processed_path
    )

def run_recommendation_system(args):
    """
    Run the recommendation system
    """
    recommender = MovieRecommender(args.processed_path)
    
    while True:
        print("\nMovie Recommendation System")
        print("==========================")
        print("Enter your search query (movie title, plot, or theme)")
        print("Type 'quit' to exit")
        
        query = input("\nQuery: ").strip()
        
        if query.lower() == 'quit':
            break
        
        if not query:
            print("Please enter a valid query")
            continue
        
        try:
            exact_matches, similar_movies = recommender.recommend_movies(
                query,
                n=args.num_recommendations
            )
            
            print("\nResults:")
            print(format_recommendations(exact_matches, similar_movies))
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Run initial setup (data collection and preprocessing)'
    )
    parser.add_argument(
        '--movies-path',
        default='../data/movies.csv',
        help='Path to save/load movies data'
    )
    parser.add_argument(
        '--processed-path',
        default='../data/processed_data.pkl',
        help='Path to save/load processed data'
    )
    parser.add_argument(
        '--num-recommendations',
        type=int,
        default=10,
        help='Number of recommendations to show'
    )
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(args.movies_path), exist_ok=True)
    
    if args.setup:
        print("Running initial setup...")
        print("Collecting movie data...")
        setup_data_collection(args)
        print("Preprocessing data...")
        setup_preprocessing(args)
        print("Setup complete!")
    
    if not os.path.exists(args.processed_path):
        print("Processed data not found. Running setup...")
        if not os.path.exists(args.movies_path):
            setup_data_collection(args)
        setup_preprocessing(args)
    
    run_recommendation_system(args)

if __name__ == "__main__":
    main() 