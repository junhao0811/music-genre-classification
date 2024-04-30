# 🎧 Music Genre Classification & Movie Clustering

Machine learning in Python on two real-world datasets: predicting a song's **genre from its audio features**, and discovering **natural groupings of top movies** with clustering.

---

## 🎵 Part 1: Music genre classifier

Predicts which of **10 genres** a song belongs to from **26,000 tracks** and their Spotify-style audio features: popularity, acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence and duration.

- Feature scaling with `StandardScaler`
- Trained and tuned two models: **Support Vector Machine** and **Random Forest**
- Compared them on accuracy, macro-averaged precision and recall (sensitivity)
- Used the best model (SVM) to predict genres for a separate, unlabelled set of songs

## 🎬 Part 2: Movie clustering

Uses the [IMDb Top 2000 Movies](https://www.kaggle.com/datasets/prishasawhney/imdb-dataset-top-2000-movies) dataset to find groups of similar films.

- Cleaned missing data
- **K-means clustering** (k = 4) on IMDb rating and box-office gross
- Visualised the clusters and interpreted them. For example, the highest-rated cluster also tends to earn the most at the box office

---

## 📁 Files

| File | Contents |
|---|---|
| `music_genre_and_movie_clustering.ipynb` | The full analysis notebook |
| `music_genre_train.csv` | Labelled songs used for training and testing |
| `music_genre_unlabelled.csv` | Unlabelled songs to predict |
| `predictions_v*.csv` | Model predictions for the unlabelled songs |
| `imdb_top_2000_movies.csv` | Movie dataset for clustering |

## 🚀 Run it
```bash
pip install pandas scikit-learn matplotlib jupyter
jupyter notebook music_genre_and_movie_clustering.ipynb
```

## 🛠️ Tech
Python · pandas · scikit-learn (SVM, Random Forest, K-means) · Matplotlib
