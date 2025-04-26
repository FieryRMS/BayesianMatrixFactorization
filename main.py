import os
import pickle
import random
from abc import ABC, abstractmethod
from csv import DictReader
from random import shuffle
from typing import TypedDict

import matplotlib.pyplot as plt
import torch
from pydantic import BaseModel
from torch import sigmoid
from tqdm import tqdm, trange

# File path to the ratings dataset
DATA_PATH = "data/ratings.csv"

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {dev}")

# Number of latent dimensions (K), MCMC samples, and rating scale
K = 5
RATINGS = torch.arange(0.5, 5.5, 0.5, device=dev)
PROPOSAL_STD = 0.002
NUM_SAMPLES = 200000
BURNIN = 1000
THIN = 40
ROLLING_WINDOW = 200
LOG_AFTER = 200
SIGMA2 = 0.1  # Likelihood noise variance
TEST_SIZE = 0.1  # Fraction of data to use for testing


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
    mn = torch.min(RATINGS)
    mx = torch.max(RATINGS)

    matrix = torch.full((num_users, num_movies), -1.0, dtype=torch.float32)
    for row in data:
        norm_rating = (row.rating - mn) / (mx - mn)  # Normalize rating to [0, 1]
        matrix[row.userId - 1, row.movieId - 1] = float(norm_rating)
    return matrix.to(dev)


def log_likelihood(matrix: torch.Tensor, U: torch.Tensor, V: torch.Tensor):
    """
    Computes the log-likelihood of observed ratings given latent vectors.
    """
    pred = sigmoid(U @ V.T)
    observed = matrix >= 0
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


def predict(preds: torch.Tensor):
    mn = float(torch.min(RATINGS))
    mx = float(torch.max(RATINGS))
    preds = preds * (mx - mn) + mn
    diffs = torch.abs(RATINGS - preds.unsqueeze(1))
    closest = torch.argmin(diffs, dim=1)
    return RATINGS[closest]


def compute_rmse(test_matrix: torch.Tensor, pred_matrix: torch.Tensor):
    observed = test_matrix >= 0
    rmse = torch.sqrt(
        ((test_matrix[observed] - predict(pred_matrix[observed])) ** 2).mean()
    )
    return rmse.item()


class Sampler(ABC):
    def __init__(self, matrix: torch.Tensor):
        self.matrix = matrix
        self.N, self.M = matrix.shape
        self.U: torch.Tensor = torch.randn((self.N, K), device=dev)
        self.V: torch.Tensor = torch.randn((self.M, K), device=dev)

    @abstractmethod
    def sample(self):
        """
        Sample from the posterior distribution.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class MetropolisHastingsSampler(Sampler):
    def __init__(self, matrix: torch.Tensor, proposal_std: float = PROPOSAL_STD):
        super().__init__(matrix)
        self.proposal_std = proposal_std
        self.cur_log_post = log_posterior(self.matrix, self.U, self.V)
        self.total_samples = 0
        self.accepted_samples = 0

    def sample(self):
        self.total_samples += 1

        # Proposal step: sample new candidate latent vectors (z*) from proposal q(z* | z)
        U_prop = self.U + torch.randn_like(self.U) * self.proposal_std
        V_prop = self.V + torch.randn_like(self.V) * self.proposal_std

        # Evaluate posterior of proposed state
        prop_log_post = log_posterior(self.matrix, U_prop, V_prop)

        # Compute Metropolis-Hastings acceptance ratio A(z*, z)
        accept_log_ratio = prop_log_post - self.cur_log_post
        # Accept proposal with probability min(1, A(z*, z))
        if torch.log(torch.rand(1, device=dev)) < accept_log_ratio:
            self.U, self.V = U_prop, V_prop
            self.cur_log_post = prop_log_post
            self.accepted_samples += 1

    def __str__(self) -> str:
        return f"[MH Sampler] Log posterior: {self.cur_log_post.item():.2f}. Accepted: {self.accepted_samples}, Rate: {self.accepted_samples / (self.total_samples):.2f}"


def run_simulation(
    matrix: torch.Tensor,
    num_samples: int = NUM_SAMPLES,
    burnin: int = BURNIN,
    thin: int = THIN,
    *,
    test_matrix: torch.Tensor | None = None,
):
    N, M = matrix.shape
    errors: list[tuple[float, int]] = []

    pred_matrix = torch.zeros((N, M), device=dev)
    samples: int = 0
    sampler: Sampler = MetropolisHastingsSampler(matrix)

    for i in trange(num_samples):
        try:
            sampler.sample()
            # Store avg sample
            if i >= burnin and i % thin == 0:
                samples += 1
                pred_matrix = (
                    pred_matrix * (samples - 1) / samples
                    + sigmoid(sampler.U @ sampler.V.T) / samples
                )

            if i % LOG_AFTER == 0:
                rmse: float | None = None
                if samples > 0 and test_matrix is not None:
                    rmse = compute_rmse(test_matrix, pred_matrix)
                    errors.append((rmse, i))
                tqdm.write(
                    f"[Sample {i}] {sampler}"
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


def init():
    if os.path.exists("config.pkl"):
        with open("config.pkl", "rb") as f:
            config: Config = pickle.load(f)
            random.setstate(config["random_state"])
            torch.set_rng_state(config["rng_state"])
            torch.cuda.set_rng_state(config["cuda_rng_state"])
    else:
        config = Config(
            random_state=random.getstate(),
            rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(),
        )
        with open("config.pkl", "wb") as f:
            pickle.dump(config, f)


def main():
    init()
    data = load_data(DATA_PATH)
    shuffle(data)
    test = data[: int(len(data) * TEST_SIZE)]
    train = data[int(len(data) * TEST_SIZE) :]

    num_movies = max(row.movieId for row in data)
    num_users = max(row.userId for row in data)
    train_matrix = create_matrix(train, num_users, num_movies)
    test_matrix = create_matrix(test, num_users, num_movies)

    tqdm.write(
        f"Train size: {len(train)}, Test size: {len(test)}, matrix size: {train_matrix.shape}"
    )

    errors: list[tuple[float, int]] = []

    if not os.path.exists("errors.pt"):
        errors = run_simulation(train_matrix, test_matrix=test_matrix)
        if errors:
            torch.save(errors, "errors.pt")  # type: ignore
    else:
        errors = torch.load("errors.pt")  # type: ignore

    # plot errors
    if errors:
        errs = torch.tensor(errors)
        plt.plot(errs[:, 1], errs[:, 0])  # type: ignore
        plt.xlabel("Sample")  # type: ignore
        plt.ylabel("RMSE")  # type: ignore
        plt.title("RMSE over samples")  # type: ignore
        plt.show()  # type: ignore


if __name__ == "__main__":
    main()
