from csv import DictReader

import numpy as np
from pydantic import BaseModel

DATA_PATH = "data/ratings.csv"


class MovieRating(BaseModel):
    """
    TypedDict for movie ratings.
    """

    userId: int
    movieId: int
    rating: float


def load_data(file_name: str):
    """
    Load data from a file.
    :param file_name: Name of the file to load data from.
    :return: Loaded data.
    """
    with open(file_name, "r") as file:
        data = DictReader(file)
        return [MovieRating.model_validate(row) for row in data]


def create_matrix(data: list[MovieRating]):
    """
    Create a normalized [0,1] matrix from the data.
    -1 is used for missing values.

    :param data: Data to create the matrix from.
    :return: Matrix created from the data.
    """
    num_users = max(row.userId for row in data)
    num_movies = max(row.movieId for row in data)
    mn = min(row.rating for row in data)
    mx = max(row.rating for row in data)

    matrix = np.full((num_users, num_movies), -1, dtype=float)

    for row in data:
        matrix[row.userId - 1, row.movieId - 1] = (row.rating - mn) / (mx - mn)
    return matrix


def main():
    MATRIX = create_matrix(load_data(DATA_PATH))
    print(f"(UserIDs, MovieIDs): {MATRIX.shape}")


if __name__ == "__main__":
    main()
