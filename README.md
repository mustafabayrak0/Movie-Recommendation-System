# Movie Recommendation System

This project implements a movie recommendation system using various techniques such as collaborative filtering, association rule mining, k-means clustering, and singular value decomposition (SVD).

# Installing MovieLens Dataset

You need to download the required datasets some of the notebooks use 25m version while some uses 100k acording to the complexity.

Also for the demo latest-small is used.

https://grouplens.org/datasets/movielens/

## Installation

To run this project, ensure you have the necessary libraries installed:

- Python 3.10
- Pandas 2.0.1
- NumPy 1.23.3
- Scikit-learn 1.2.2
- MLxtend 0.23.0
- Keras 2.10.0
- Surprise 0.1
- SciPy 1.10.1
- NetworkX 3.1

Install these dependencies using pip:

```bash
pip install pandas==2.0.1 numpy==1.23.3 scikit-learn==1.2.2 mlxtend==0.23.0 keras==2.10.0 surprise==0.1 scipy==1.10.1 networkx==3.1


How to Run

    Clone this repository.
    Install the required libraries as mentioned above.
    Set up the necessary environment.
    Run the main Python file.

```bash
python demo.py

Usage

Upon running the code, users will be prompted to rate a set of movies. These ratings are then used to generate personalized movie recommendations. The system provides options for different recommendation techniques:

    KNN Recommendation
    Collaborative Filtering
    Appriori Rule
    Singular Value Decomposition

Users can select the desired option to receive movie recommendations.

Select the desired option to get movie recommendations using the respective technique.
