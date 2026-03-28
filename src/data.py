"""
Multi-agent consensus simulation environment.

Generates episodes where N agents with random initial states must reach
consensus (converge to the mean of their initial states) via distributed control.
"""
import torch


class ConsensusEnv:
    """Simple single-integrator multi-agent consensus environment.

    Dynamics: x_{t+1} = x_t + dt * u_t
    Goal: all agents converge to mean(x_0).

    Args:
        n_agents: Number of agents.
        state_dim: Dimension of each agent's state.
        dt: Integration timestep.
    """

    def __init__(self, n_agents: int = 6, state_dim: int = 1, dt: float = 0.1):
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.dt = dt

    def reset(self, batch_size: int = 1, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Sample random initial states and return them.

        Returns:
            x0: shape (batch_size, n_agents, state_dim), sampled from N(0, 1).
        """
        x0 = torch.randn(batch_size, self.n_agents, self.state_dim, device=device)
        return x0

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Apply control and return next state.

        Args:
            x: Current states (batch_size, n_agents, state_dim).
            u: Control inputs (batch_size, n_agents, control_dim).

        Returns:
            x_next: Next states (batch_size, n_agents, state_dim).
        """
        return x + self.dt * u

    @staticmethod
    def consensus_target(x0: torch.Tensor) -> torch.Tensor:
        """Compute consensus target (mean of initial states).

        Args:
            x0: Initial states (batch_size, n_agents, state_dim).

        Returns:
            target: shape (batch_size, 1, state_dim), broadcast-ready.
        """
        return x0.mean(dim=1, keepdim=True)
