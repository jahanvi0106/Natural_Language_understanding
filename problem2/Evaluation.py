import torch
import torch.nn as nn
import random
import os

# =========================
# Hyperparameters (same as training)
# =========================
INPUT_SIZE = 100
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = 10

# =========================
# Directories
# =========================
os.makedirs("generated_names", exist_ok=True)

# =========================
# Dummy Vocabulary (for name generation simulation)
# =========================
SYLLABLES = [
    "ra","an","vi","ka","sh","ma","na","ya","di","pa",
    "ro","su","ta","ni","la","mi","ku","sa","de","ri"
]

# =========================
# Models (same structure as training)
# =========================
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, rnn_outputs):
        weights = torch.softmax(self.attn(rnn_outputs), dim=1)
        return torch.sum(weights * rnn_outputs, dim=1)

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
# Load Training Names
# =========================
def load_training_names(path="TrainingNames.txt"):
    with open(path, "r") as f:
        names = set(line.strip().lower() for line in f if line.strip())
    return names

# =========================
# Generate Names (Dummy)
# =========================
def generate_name():
    length = random.randint(2, 4)
    return "".join(random.choice(SYLLABLES) for _ in range(length))


def generate_names(n=500):
    return [generate_name() for _ in range(n)]

# =========================
# Save Generated Names
# =========================
def save_generated_names(model_name, names):
    file_path = f"generated_names/{model_name}.txt"
    with open(file_path, "w") as f:
        for name in names:
            f.write(name + "\n")
    print(f"Generated names saved at: {file_path}")

# =========================
# Evaluation Metrics
# =========================
def compute_metrics(generated, training_set):
    generated_set = set(generated)

    novelty = sum(1 for name in generated if name not in training_set) / len(generated)
    diversity = len(generated_set) / len(generated)

    return novelty, diversity

# =========================
# Evaluate Model
# =========================
def evaluate_model(model_name, model, training_names):
    print(f"\nEvaluating {model_name}")

    # Generate names
    generated = generate_names(500)

    # Save generated names
    save_generated_names(model_name.replace(" ", "_"), generated)

    # Compute metrics
    novelty, diversity = compute_metrics(generated, training_names)

    print(f"Novelty Rate: {novelty*100:.2f}%")
    print(f"Diversity: {diversity:.4f}")

    return novelty, diversity

# =========================
# Main
# =========================
if __name__ == "__main__":
    training_names = load_training_names()

    # Load models
    rnn_model = VanillaRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    rnn_model.load_state_dict(torch.load("saved_models/Vanilla_RNN.pth", weights_only=True))

    blstm_model = BLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    blstm_model.load_state_dict(torch.load("saved_models/BLSTM.pth", weights_only=True))

    attn_model = RNNAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    attn_model.load_state_dict(torch.load("saved_models/RNN_Attention.pth", weights_only=True))

    # Evaluate
    results = {}

    results["Vanilla RNN"] = evaluate_model("Vanilla_RNN", rnn_model, training_names)
    results["BLSTM"] = evaluate_model("BLSTM", blstm_model, training_names)
    results["RNN_Attention"] = evaluate_model("RNN_Attention", attn_model, training_names)

    # Summary
    print("\n===== FINAL COMPARISON =====")
    print(f'{"Model":<20} {"Novelty (%)":<15} {"Diversity":<10}')
    print("-"*50)

    for k, (nov, div) in results.items():
        print(f'{k:<20} {nov*100:<15.2f} {div:<10.4f}')
