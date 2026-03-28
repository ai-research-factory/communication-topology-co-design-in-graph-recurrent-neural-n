"""
GRNN Controller with learnable adjacency matrix for multi-agent consensus.

Implements the Graph Recurrent Neural Network from:
"Communication Topology Co-Design in Graph Recurrent Neural Network Based Distributed Control"
"""
import torch
import torch.nn as nn


class GRNNController(nn.Module):
    """Graph Recurrent Neural Network controller with learnable communication topology.

    The adjacency matrix A is a learnable parameter. L1 regularization on A
    during training encourages sparse communication graphs.

    Args:
        n_agents: Number of agents in the system.
        state_dim: Dimension of each agent's state vector.
        hidden_dim: Dimension of the GRU hidden state.
        control_dim: Dimension of the control output per agent.
    """

    def __init__(self, n_agents: int, state_dim: int = 1, hidden_dim: int = 16,
                 control_dim: int = 1):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.control_dim = control_dim

        # Learnable adjacency matrix (communication topology)
        self.A = nn.Parameter(torch.randn(n_agents, n_agents) * 0.1)

        # GNN message-passing layer: transforms aggregated neighbor messages
        self.msg_fc = nn.Linear(state_dim, hidden_dim)

        # GRU cell for temporal dynamics per agent
        self.gru = nn.GRUCell(state_dim + hidden_dim, hidden_dim)

        # Control output layer
        self.control_fc = nn.Linear(hidden_dim, control_dim)

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize hidden state to zeros. Shape: (batch_size * n_agents, hidden_dim)."""
        return torch.zeros(batch_size * self.n_agents, self.hidden_dim,
                           device=self.A.device)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One step of the GRNN controller.

        Args:
            x: Agent states, shape (batch_size, n_agents, state_dim).
            h: Hidden states, shape (batch_size * n_agents, hidden_dim).

        Returns:
            u: Control actions, shape (batch_size, n_agents, control_dim).
            h_new: Updated hidden states, shape (batch_size * n_agents, hidden_dim).
        """
        batch_size = x.shape[0]

        # Soft adjacency via sigmoid to keep weights in [0, 1]
        A_soft = torch.sigmoid(self.A)  # (n_agents, n_agents)

        # GNN message passing: aggregate neighbor states weighted by A
        # x: (B, N, state_dim), A_soft: (N, N)
        # agg: (B, N, state_dim) — weighted sum of neighbor states
        agg = torch.matmul(A_soft, x)  # broadcast over batch

        # Transform aggregated messages
        msg = self.msg_fc(agg)  # (B, N, hidden_dim)

        # Reshape for GRU: flatten batch and agent dims
        x_flat = x.reshape(batch_size * self.n_agents, self.state_dim)
        msg_flat = msg.reshape(batch_size * self.n_agents, self.hidden_dim)

        # GRU input: concatenation of own state and aggregated message
        gru_input = torch.cat([x_flat, msg_flat], dim=-1)  # (B*N, state_dim + hidden_dim)
        h_new = self.gru(gru_input, h)  # (B*N, hidden_dim)

        # Control output
        u_flat = self.control_fc(h_new)  # (B*N, control_dim)
        u = u_flat.reshape(batch_size, self.n_agents, self.control_dim)

        return u, h_new
