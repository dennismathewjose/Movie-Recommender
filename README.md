# Movie Recommendation System using NLP

This project implements a plot-based movie recommendation system using Natural Language Processing (NLP) techniques, specifically leveraging Sentence-BERT for measuring movie similarities based on plot summaries.

## Features

- Search movies by theme, plot, title, director or genre
- Get exact matches and similar movies based on plot content
- View similarity scores for recommended movies
- Display top 10 similar movies with their genres along with user-rating
- Process and analyze a large dataset of 1000+ movies

## Project Structure

```
.
├── data/               # Dataset storage
├── docs/              # Project documentation
├── notebooks/         # Jupyter notebooks for analysis
├── src/              # Source code
└── requirements.txt   # Project dependencies
```

## Setup and Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd movie-recommendation-system
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preparation:
   - The system uses movie data from [dataset source]
   - Run data preprocessing scripts in the notebooks directory
   - Processed data will be saved in the data directory

2. Running the System:
   - Execute the main script:
   ```bash
   python src/main.py
   ```
   - Follow the prompts to search for movies

## Model Architecture

The system uses Sentence-BERT (SBERT) for generating embeddings of movie plot summaries. SBERT is chosen for its:
- Superior performance in semantic similarity tasks
- Efficient computation of sentence embeddings
- Ability to capture contextual information

## Documentation

Detailed documentation is available in the `docs` directory, covering:
- Data collection and preprocessing methodology
- Model architecture and training process
- Evaluation metrics and results
- API documentation (if applicable)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
