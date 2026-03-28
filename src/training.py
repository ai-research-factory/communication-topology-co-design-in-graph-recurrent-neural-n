"""
Training loop for GRNN-based distributed consensus controller.

Implements the composite loss: performance_loss + lambda * L1_norm(A)
where performance_loss measures deviation from consensus and L1_norm
encourages sparse communication topology.
"""
import torch
from src.model import GRNNController
from src.data import ConsensusEnv


def compute_performance_loss(trajectory: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Compute performance loss over a trajectory.

    Sum of squared errors between agent states and consensus target
    across all timesteps.

    Args:
        trajectory: List of state tensors, each (batch_size, n_agents, state_dim).
        target: Consensus target (batch_size, 1, state_dim).

    Returns:
        Scalar loss tensor.
    """
    loss = torch.tensor(0.0, device=trajectory[0].device)
    for x in trajectory:
        loss = loss + ((x - target) ** 2).sum(dim=(1, 2)).mean()
    return loss / len(trajectory)


def run_episode(model: GRNNController, env: ConsensusEnv, T: int,
                batch_size: int = 32, device: torch.device = torch.device("cpu")
                ) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Run a single simulation episode.

    Args:
        model: GRNN controller.
        env: Consensus environment.
        T: Number of timesteps to unroll.
        batch_size: Number of parallel episodes.
        device: Torch device.

    Returns:
        trajectory: List of T+1 state tensors.
        target: Consensus target from initial states.
    """
    x = env.reset(batch_size=batch_size, device=device)
    target = ConsensusEnv.consensus_target(x)
    h = model.init_hidden(batch_size=batch_size)

    trajectory = [x]
    for _ in range(T):
        u, h = model(x, h)
        x = env.step(x, u)
        trajectory.append(x)

    return trajectory, target


def train_epoch(model: GRNNController, env: ConsensusEnv, optimizer: torch.optim.Optimizer,
                n_episodes: int = 16, T: int = 50, batch_size: int = 32,
                lam: float = 0.01, device: torch.device = torch.device("cpu")
                ) -> dict:
    """Train for one epoch (multiple episodes).

    Args:
        model: GRNN controller.
        env: Consensus environment.
        optimizer: Optimizer for model parameters (including A).
        n_episodes: Number of episodes per epoch.
        T: Timesteps per episode.
        batch_size: Batch size per episode.
        lam: L1 regularization strength on adjacency matrix.
        device: Torch device.

    Returns:
        Dict with epoch statistics: avg_loss, avg_perf_loss, l1_norm.
    """
    model.train()
    total_loss = 0.0
    total_perf_loss = 0.0

    for _ in range(n_episodes):
        trajectory, target = run_episode(model, env, T, batch_size, device)

        perf_loss = compute_performance_loss(trajectory, target)
        l1_norm = torch.norm(model.A, p=1)
        loss = perf_loss + lam * l1_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_perf_loss += perf_loss.item()

    n = n_episodes
    return {
        "avg_loss": total_loss / n,
        "avg_perf_loss": total_perf_loss / n,
        "l1_norm": torch.norm(model.A, p=1).item(),
    }
