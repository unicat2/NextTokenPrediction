import matplotlib.pyplot as plt
import numpy as np


def load_loss_data(file_path):
    epochs, losses = [], []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line or ", " not in line:
                continue
            try:
                epoch, loss = line.split(", ")
                epoch = int(epoch.split()[1])
                loss = float(loss.split(": ")[1])
                epochs.append(epoch)
                losses.append(loss)
            except (IndexError, ValueError):
                print(f"Skipping malformed line: {line}")
    return np.array(epochs), np.array(losses)


def calculate_perplexity(losses):
    return np.exp(losses)


paths = {
    "Transformer": {
        "train": "./Transformer/train_log.txt",
        "eval": "./Transformer/eval_log.txt"
    },
    "FFN": {
        "train": "./FFN/train_log.txt",
        "eval": "./FFN/eval_log.txt"
    },
    "LSTM": {
        "train": "./LSTM/train_log.txt",
        "eval": "./LSTM/eval_log.txt"
    }
}

colors = ['tab:blue', 'tab:orange', 'tab:green']
linestyles = ['-', '--', ':']
network_names = ["Transformer", "FFN", "LSTM"]

# Training Loss
plt.figure(figsize=(8, 6))
for idx, network in enumerate(network_names):
    train_epochs, train_losses = load_loss_data(paths[network]["train"])
    plt.plot(train_epochs, train_losses, label=f"{network} - Train Loss", color=colors[idx],
             linestyle=linestyles[idx], linewidth=2, alpha=0.8)

plt.title("Training Loss", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig("training_loss_comparison.png", dpi=300)
plt.show()

# Evaluation Loss
plt.figure(figsize=(8, 6))
for idx, network in enumerate(network_names):
    eval_epochs, eval_losses = load_loss_data(paths[network]["eval"])
    plt.plot(eval_epochs, eval_losses, label=f"{network} - Eval Loss", color=colors[idx],
             linestyle=linestyles[idx], linewidth=2, alpha=0.8)

plt.title("Evaluation Loss", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig("evaluation_loss_comparison.png", dpi=300)
plt.show()

# Perplexity
plt.figure(figsize=(8, 6))
for idx, network in enumerate(network_names):
    eval_epochs, eval_losses = load_loss_data(paths[network]["eval"])
    eval_perplexity = calculate_perplexity(eval_losses)
    plt.plot(eval_epochs, eval_perplexity, label=f"{network} - Eval Perplexity", color=colors[idx],
             linestyle=linestyles[idx], linewidth=2, alpha=0.8)

plt.title("Evaluation Perplexity", fontsize=16, fontweight='bold')
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Perplexity", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig("evaluation_perplexity_comparison.png", dpi=300)
plt.show()

