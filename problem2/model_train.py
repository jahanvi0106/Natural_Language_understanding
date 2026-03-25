import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# =========================
# Hyperparameters
# =========================
INPUT_SIZE = 100
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = 10
LEARNING_RATE = 0.001
EPOCHS = 40

# Create directory
os.makedirs("saved_models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# =========================
# Utility: Count parameters
# =========================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =========================
# 1. Vanilla RNN
# =========================
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)

# =========================
# 2. Bidirectional LSTM
# =========================
class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# =========================
# 3. Attention Mechanism
# =========================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, rnn_outputs):
        weights = torch.softmax(self.attn(rnn_outputs), dim=1)
        context = torch.sum(weights * rnn_outputs, dim=1)
        return context

class RNNAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        context = self.attention(out)
        return self.fc(context)

# =========================
# Training Function with Loss Tracking + Saving
# =========================
def train_model(model, name):
    print(f"\nTraining {name}")
    print(f"Trainable Parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    x = torch.randn(32, 20, INPUT_SIZE)
    y = torch.randint(0, OUTPUT_SIZE, (32,))

    losses = []

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Save model
    save_path = f"saved_models/{name.replace(' ', '_').replace('+','')}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at: {save_path}")

    return losses

# =========================
# Plot Function (Saved)
# =========================
def plot_losses(loss_dict):
    plt.figure()
    for name, losses in loss_dict.items():
        plt.plot(losses, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Loss Comparison")
    plt.legend()
    plt.grid()

    plot_path = "plots/loss_comparison.png"
    plt.savefig(plot_path)
    print(f"Plot saved at: {plot_path}")

    plt.show()

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    rnn_model = VanillaRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    blstm_model = BLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    attn_model = RNNAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

    print("\n===== MODEL CONFIGURATION =====")
    print(f"Hidden Size: {HIDDEN_SIZE}")
    print(f"Layers: {NUM_LAYERS}")
    print(f"Learning Rate: {LEARNING_RATE}")

    loss_dict = {}

    loss_dict["Vanilla RNN"] = train_model(rnn_model, "Vanilla_RNN")
    loss_dict["BLSTM"] = train_model(blstm_model, "BLSTM")
    loss_dict["RNN + Attention"] = train_model(attn_model, "RNN_Attention")

    plot_losses(loss_dict)
