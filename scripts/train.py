#!/usr/bin/env python3
"""
CLI for training the GRNN consensus controller.

Usage:
    python3 scripts/train.py --epochs 50 --lr 0.003 --lam 0.01 --n-agents 6
"""
import argparse
import json
import os
import sys

import torch

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import GRNNController
from src.data import ConsensusEnv
from src.training import train_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Train GRNN consensus controller")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--lam", type=float, default=0.01, help="L1 regularization strength")
    parser.add_argument("--n-agents", type=int, default=6, help="Number of agents")
    parser.add_argument("--state-dim", type=int, default=1, help="State dimension per agent")
    parser.add_argument("--hidden-dim", type=int, default=16, help="GRU hidden dimension")
    parser.add_argument("--T", type=int, default=50, help="Timesteps per episode")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--n-episodes", type=int, default=16, help="Episodes per epoch")
    parser.add_argument("--dt", type=float, default=0.1, help="Simulation timestep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", type=str, default="reports/cycle_2", help="Log output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize environment and model
    env = ConsensusEnv(n_agents=args.n_agents, state_dim=args.state_dim, dt=args.dt)
    model = GRNNController(
        n_agents=args.n_agents,
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        control_dim=args.state_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Config: epochs={args.epochs}, lr={args.lr}, lam={args.lam}, "
          f"N={args.n_agents}, T={args.T}, batch={args.batch_size}")
    print("-" * 70)

    # Create output directory
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "training_log.txt")

    history = []
    with open(log_path, "w") as log_file:
        header = f"{'Epoch':>5} | {'Total Loss':>12} | {'Perf Loss':>12} | {'L1 Norm(A)':>12}"
        print(header)
        log_file.write(header + "\n")
        log_file.write("-" * 70 + "\n")

        for epoch in range(1, args.epochs + 1):
            stats = train_epoch(
                model=model,
                env=env,
                optimizer=optimizer,
                n_episodes=args.n_episodes,
                T=args.T,
                batch_size=args.batch_size,
                lam=args.lam,
                device=device,
            )
            history.append(stats)

            line = (f"{epoch:5d} | {stats['avg_loss']:12.6f} | "
                    f"{stats['avg_perf_loss']:12.6f} | {stats['l1_norm']:12.6f}")
            print(line)
            log_file.write(line + "\n")

    # Save model checkpoint
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/grnn_controller.pt")
    print(f"\nModel saved to models/grnn_controller.pt")

    # Generate metrics.json
    final = history[-1]
    initial = history[0]
    metrics = {
        "sharpeRatio": 0.0,
        "annualReturn": 0.0,
        "maxDrawdown": 0.0,
        "hitRate": 0.0,
        "totalTrades": 0,
        "transactionCosts": {"feeBps": 10, "slippageBps": 5, "netSharpe": 0.0},
        "walkForward": {"windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0},
        "customMetrics": {
            "final_total_loss": round(final["avg_loss"], 6),
            "final_perf_loss": round(final["avg_perf_loss"], 6),
            "final_l1_norm": round(final["l1_norm"], 6),
            "initial_total_loss": round(initial["avg_loss"], 6),
            "initial_perf_loss": round(initial["avg_perf_loss"], 6),
            "initial_l1_norm": round(initial["l1_norm"], 6),
            "loss_reduction_pct": round(
                (1 - final["avg_loss"] / initial["avg_loss"]) * 100, 2
            ) if initial["avg_loss"] > 0 else 0.0,
            "n_agents": args.n_agents,
            "epochs": args.epochs,
            "lambda": args.lam,
            "learning_rate": args.lr,
        },
    }
    metrics_path = os.path.join(args.log_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
