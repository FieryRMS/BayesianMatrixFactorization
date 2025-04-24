import os
import pickle
import random
from abc import ABC, abstractmethod
from csv import DictReader
from random import shuffle
from typing import Any, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import BaseModel
from tqdm import tqdm, trange

# File path to the ratings dataset
DATA_PATH = "data/ratings.csv"

# Number of latent dimensions (K), MCMC samples, and rating scale
K = 5
RATINGS = np.arange(0.5, 5.5, 0.5)
PROPOSAL_STD = 0.002
NUM_SAMPLES = 200000
BURNIN = 4000
THIN = 10
ROLLING_WINDOW = 200
LOG_AFTER = 200
SIGMA2 = 0.1  # Likelihood noise variance
TEST_SIZE = 0.1  # Fraction of data to use for testing


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {dev}")


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


def create_matrix(data: list[MovieRating], num_users: int, num_movies: int):
    """
    Create a normalized [0,1] matrix from the data.
    -1 is used for missing values.

    :param data: Data to create the matrix from.
    :return: Matrix created from the data.
    """
    mn = np.min(RATINGS)
    mx = np.max(RATINGS)

    matrix = torch.full((num_users, num_movies), -1.0, dtype=torch.float32)
    for row in data:
        norm_rating = (row.rating - mn) / (mx - mn)  # Normalize rating to [0, 1]
        matrix[row.userId - 1, row.movieId - 1] = float(norm_rating)
    return matrix.to(dev)


def sigmoid(x: torch.Tensor):
    return torch.sigmoid(x)


def log_likelihood(matrix: torch.Tensor, U: torch.Tensor, V: torch.Tensor):
    """
    Computes the log-likelihood of observed ratings given latent vectors.
    """
    pred = sigmoid(U @ V.T)
    observed = matrix != -1
    error = matrix[observed] - pred[observed]
    return -0.5 * torch.sum(error**2) / SIGMA2


def log_prior(U: torch.Tensor, V: torch.Tensor):
    """
    Computes log prior assuming standard normal priors on latent vectors.
    """
    return -0.5 * (U**2).sum() - 0.5 * (V**2).sum()


def log_posterior(matrix: torch.Tensor, U: torch.Tensor, V: torch.Tensor):
    """
    Log posterior ~ log_likelihood + log_prior
    """
    return log_likelihood(matrix, U, V) + log_prior(U, V)


def predict(
    samples: list[tuple[torch.Tensor, torch.Tensor]], user_id: int, movie_id: int
):
    preds: list[float] = []
    for U, V in samples:
        u = U[user_id]
        v = V[movie_id]
        pred = sigmoid(u @ v)
        preds.append(pred.item())

    mn = np.min(RATINGS)
    mx = np.max(RATINGS)

    avg_pred = np.mean(preds) * (mx - mn) + mn
    # Rescale back to original ratings
    rating_candidates = torch.tensor(RATINGS, device=dev)
    diffs = torch.abs(rating_candidates - float(avg_pred))
    closest = torch.argmin(diffs).item()
    return closest


def predict_user(
    samples: list[tuple[torch.Tensor, torch.Tensor]],
    user_id: int,
    movie_mask: torch.Tensor,
):
    preds: list[torch.Tensor] = []
    for U, V in samples:
        u = U[user_id]
        v = V[movie_mask]
        pred = sigmoid(u @ v.T)
        preds.append(pred)

    mn = float(np.min(RATINGS))
    mx = float(np.max(RATINGS))
    # avg_pred = torch.mean(torch.stack(preds), dim=0) * (mx - mn) + mn
    avg_pred = torch.stack(preds).mean(dim=0) * (mx - mn) + mn
    # Rescale back to original ratings
    rating_candidates = torch.tensor(RATINGS, device=dev)
    diffs = torch.abs(rating_candidates - avg_pred.unsqueeze(1))
    closest = torch.argmin(diffs, dim=1)
    return closest


def predict_all(
    samples: list[tuple[torch.Tensor, torch.Tensor]],
    mask: torch.Tensor,
):
    preds: list[torch.Tensor] = []
    for U, V in samples:
        pred = sigmoid(U @ V.T)
        preds.append(pred[mask])

    mn = float(np.min(RATINGS))
    mx = float(np.max(RATINGS))
    avg_pred = torch.stack(preds).mean(dim=0) * (mx - mn) + mn
    # Rescale back to original ratings
    rating_candidates = torch.tensor(RATINGS, device=dev)
    diffs = torch.abs(rating_candidates - avg_pred.unsqueeze(1))
    closest = torch.argmin(diffs, dim=1)
    return closest


def unnormalize(preds: torch.Tensor):
    mn = float(np.min(RATINGS))
    mx = float(np.max(RATINGS))
    preds = preds * (mx - mn) + mn
    rating_candidates = torch.tensor(RATINGS, device=dev)
    diffs = torch.abs(rating_candidates - preds.unsqueeze(1))
    closest = torch.argmin(diffs, dim=1)
    return closest


def compute_rmse(test_matrix: torch.Tensor, pred_matrix: torch.Tensor):
    observed = test_matrix != -1
    rmse = torch.sqrt(
        ((test_matrix[observed] - unnormalize(pred_matrix[observed])) ** 2).mean()
    )
    return rmse.item()


class Sampler:
    def __init__(self, matrix: torch.Tensor):
        self.N, self.M = matrix.shape
        self._U = torch.randn((self.N, K), device=dev)
        self._V = torch.randn((self.M, K), device=dev)

    @property
    def U(self):
        return self._U

    @property
    def V(self):
        return self._V

    @abstractmethod
    def sample(self):
        """
        Sample from the posterior distribution.
        """
        pass


class MetropolisHastingsSampler(Sampler):
    def __init__(self, matrix: torch.Tensor, proposal_std: float = PROPOSAL_STD):
        super().__init__(matrix)
        self.proposal_std = proposal_std
        cur_log_post = log_posterior(matrix, U, V)

    def sample(self):
        # Proposal step: sample new candidate latent vectors (z*) from proposal q(z* | z)
        U_prop = self._U + torch.randn_like(self._U) * self.proposal_std
        V_prop = self.V + torch.randn_like(self.V) * self.proposal_std

        # Evaluate posterior of proposed state
        prop_log_post = log_posterior(matrix, U_prop, V_prop)

        # Compute Metropolis-Hastings acceptance ratio A(z*, z)
        accept_log_ratio = prop_log_post - cur_log_post
        # Accept proposal with probability min(1, A(z*, z))
        if torch.log(torch.rand(1, device=dev)) < accept_log_ratio:
            U, V = U_prop, V_prop  # type: ignore
            cur_log_post = prop_log_post
            accepted_samples += 1


def metropolis_hastings(
    matrix: torch.Tensor,
    num_samples: int = NUM_SAMPLES,
    proposal_std: float = PROPOSAL_STD,
    burnin: int = BURNIN,
    thin: int = THIN,
    *,
    test_matrix: torch.Tensor | None = None,
):
    N, M = matrix.shape
    U = torch.randn((N, K), device=dev)
    V = torch.randn((M, K), device=dev)
    errors: list[tuple[float, int]] = []

    pred_matrix = torch.zeros((N, M), device=dev)
    samples: int = 0
    accepted_samples = 0

    for i in trange(num_samples):
        try:

            # Store avg sample
            if i >= burnin and i % thin == 0:
                samples += 1
                pred_matrix = (
                    pred_matrix * (samples - 1) / samples + sigmoid(U @ V.T) / samples
                )

            if i % LOG_AFTER == 0:
                rmse: float | None = None
                if samples > 0 and test_matrix is not None:
                    rmse = compute_rmse(test_matrix, pred_matrix)
                    errors.append((rmse, i))
                tqdm.write(
                    f"[Sample {i}] Log posterior: {cur_log_post.item():.2f}. Accepted: {accepted_samples}, Rate: {accepted_samples / (i + 1):.2f}"
                    + (f", RMSE: {rmse:.6f}" if rmse is not None else "")
                )
        except KeyboardInterrupt:
            tqdm.write("Keyboard interrupt. Stopping sampling.")
            return errors

    return errors


class Config(TypedDict):
    random_state: tuple[int, ...]
    rng_state: torch.Tensor
    cuda_rng_state: torch.Tensor
    numpy_state: dict[str, Any]


def init():
    if os.path.exists("config.pkl"):
        with open("config.pkl", "rb") as f:
            config: Config = pickle.load(f)
            random.setstate(config["random_state"])
            np.random.set_state(config["numpy_state"])
            torch.set_rng_state(config["rng_state"])
            torch.cuda.set_rng_state(config["cuda_rng_state"])
    else:
        config = Config(
            random_state=random.getstate(),
            rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(),
            numpy_state=np.random.get_state(),
        )
        with open("config.pkl", "wb") as f:
            pickle.dump(config, f)


def main():
    init()
    data = load_data(DATA_PATH)
    shuffle(data)
    test = data[: int(len(data) * TEST_SIZE)]
    train = data[int(len(data) * TEST_SIZE) :]
    tqdm.write(f"Train size: {len(train)}, Test size: {len(test)}")

    num_movies = max(row.movieId for row in data)
    num_users = max(row.userId for row in data)
    train_matrix = create_matrix(train, num_users, num_movies)
    test_matrix = create_matrix(test, num_users, num_movies)

    errors: list[tuple[float, int]] = []

    if not os.path.exists("samples.pt"):
        errors = metropolis_hastings(train_matrix, test_matrix=test_matrix)
        torch.save(samples, "samples.pt")  # type: ignore
        torch.save(errors, "errors.pt")  # type: ignore
    else:
        errors = torch.load("errors.pt")  # type: ignore

    # calculate rmse
    # print("FINAL RMSE: ", compute_rmse(test_matrix, samples))

    # plot errors
    if errors:
        errors = np.array(errors)
        plt.plot(errors[:, 1], errors[:, 0])
        plt.xlabel("Sample")
        plt.ylabel("RMSE")
        plt.title("RMSE over samples")
        plt.show()


if __name__ == "__main__":
    main()
