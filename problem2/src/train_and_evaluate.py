"""
TASK-1 & 2: Training and Evaluation
====================================

This script handles:
- Data preprocessing
- Model training
- Name generation
- Evaluation (novelty rate, diversity)

Usage:
    python train_and_evaluate.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm
import json
import os
from collections import Counter

from task1_models import VanillaRNN, BidirectionalLSTM, AttentionRNN

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class NameDataset(Dataset):
    """Dataset for character-level name generation"""
    
    def __init__(self, names, char_to_idx, max_len=20):
        self.names = names
        self.char_to_idx = char_to_idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        name = self.names[idx]
        
        # Add start and end tokens
        name = '<SOS>' + name + '<EOS>'
        
        # Convert to indices
        indices = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) 
                   for c in name]
        
        # Pad to max_len
        if len(indices) < self.max_len:
            indices += [self.char_to_idx['<PAD>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        # Input is all characters except last, target is all except first
        x = torch.tensor(indices[:-1], dtype=torch.long)
        y = torch.tensor(indices[1:], dtype=torch.long)
        
        return x, y


def load_and_preprocess_data(filename='TrainingNames.txt'):
    """Load names and create vocabulary"""
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    # Load names
    with open(filename, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(names)} names")
    
    # Build vocabulary
    all_chars = set()
    for name in names:
        all_chars.update(name)
    
    # Special tokens
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    vocab = special_tokens + sorted(list(all_chars))
    
    char_to_idx = {c: i for i, c in enumerate(vocab)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Characters: {' '.join([c for c in vocab if c not in special_tokens][:30])}")
    
    # Statistics
    avg_len = np.mean([len(name) for name in names])
    max_len = max(len(name) for name in names)
    min_len = min(len(name) for name in names)
    
    print(f"Average name length: {avg_len:.1f}")
    print(f"Length range: {min_len} - {max_len}")
    
    return names, char_to_idx, idx_to_char


def train_model(model, train_loader, optimizer, criterion, device, model_name):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Training {model_name}")
    
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if isinstance(model, AttentionRNN):
            output, _, _ = model(x)
        else:
            output, _ = model(x)
        
        # Compute loss
        output = output.reshape(-1, output.size(-1))
        y = y.reshape(-1)
        
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def generate_name(model, char_to_idx, idx_to_char, device, 
                  max_len=20, temperature=0.8):
    """
    Generate a single name
    
    Args:
        model: Trained model
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        device: Device to use
        max_len: Maximum name length
        temperature: Sampling temperature (higher = more random)
        
    Returns:
        Generated name string
    """
    model.eval()
    
    with torch.no_grad():
        # Start with SOS token
        current_char = char_to_idx['<SOS>']
        generated = []
        
        # Initialize hidden state
        hidden = None
        
        for _ in range(max_len):
            # Prepare input
            x = torch.tensor([[current_char]], dtype=torch.long).to(device)
            
            # Forward pass
            if isinstance(model, AttentionRNN):
                output, hidden, _ = model(x, hidden)
            else:
                output, hidden = model(x, hidden)
            
            # Get probabilities for next character
            logits = output[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=0)
            
            # Sample next character
            next_char_idx = torch.multinomial(probs, 1).item()
            
            # Check for EOS
            if idx_to_char[next_char_idx] == '<EOS>':
                break
            
            # Skip special tokens
            if idx_to_char[next_char_idx] in ['<PAD>', '<SOS>', '<UNK>']:
                continue
            
            generated.append(idx_to_char[next_char_idx])
            current_char = next_char_idx
        
        return ''.join(generated)


def generate_names(model, char_to_idx, idx_to_char, device, num_names=1000,
                   temperature=0.8):
    """Generate multiple names"""
    names = []
    attempts = 0
    max_attempts = num_names * 3
    
    pbar = tqdm(total=num_names, desc="Generating names")
    
    while len(names) < num_names and attempts < max_attempts:
        attempts += 1
        name = generate_name(model, char_to_idx, idx_to_char, device, 
                           temperature=temperature)
        
        # Basic filtering
        if 3 <= len(name) <= 15:
            names.append(name)
            pbar.update(1)
    
    pbar.close()
    
    return names


def evaluate_model(generated_names, training_names):
    """
    Evaluate generated names
    
    Args:
        generated_names: List of generated names
        training_names: List of training names
        
    Returns:
        Dictionary with evaluation metrics
    """
    training_set = set(training_names)
    
    # Novelty Rate
    novel_names = [name for name in generated_names if name not in training_set]
    novelty_rate = len(novel_names) / len(generated_names) * 100
    
    # Diversity
    unique_names = len(set(generated_names))
    diversity = unique_names / len(generated_names) * 100
    
    # Average length
    avg_length = np.mean([len(name) for name in generated_names])
    
    return {
        'total_generated': len(generated_names),
        'novel_names': len(novel_names),
        'novelty_rate': novelty_rate,
        'unique_names': unique_names,
        'diversity': diversity,
        'avg_length': avg_length
    }


def print_evaluation(metrics, model_name):
    """Print evaluation metrics"""
    print(f"\n{'='*70}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*70}")
    print(f"Total Generated:    {metrics['total_generated']}")
    print(f"Novel Names:        {metrics['novel_names']} ({metrics['novelty_rate']:.2f}%)")
    print(f"Unique Names:       {metrics['unique_names']} ({metrics['diversity']:.2f}%)")
    print(f"Average Length:     {metrics['avg_length']:.2f}")
    print(f"{'='*70}\n")


def main():
    """Main training and evaluation pipeline"""
    print("="*70)
    print("RNN NAME GENERATION - TRAINING & EVALUATION")
    print("="*70)
    
    # Hyperparameters
    EMBEDDING_DIM = 64
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {DEVICE}")
    
    # Load data
    names, char_to_idx, idx_to_char = load_and_preprocess_data('TrainingNames.txt')
    vocab_size = len(char_to_idx)
    
    # Create dataset
    max_len = max(len(name) for name in names) + 3  # +3 for SOS, EOS, and padding
    dataset = NameDataset(names, char_to_idx, max_len)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Models to train
    models_config = [
        ('Vanilla RNN', VanillaRNN(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, 
                                   NUM_LAYERS, DROPOUT)),
        ('Bidirectional LSTM', BidirectionalLSTM(vocab_size, EMBEDDING_DIM, 
                                                  HIDDEN_SIZE, NUM_LAYERS, DROPOUT)),
        ('RNN with Attention', AttentionRNN(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE,
                                            NUM_LAYERS, DROPOUT))
    ]
    
    # Train and evaluate each model
    results = {}
    
    for model_name, model in models_config:
        print(f"\n{'#'*70}")
        print(f"MODEL: {model_name}")
        print(f"{'#'*70}")
        
        model = model.to(DEVICE)
        
        # Print architecture
        info = model.get_architecture_info()
        print(f"\nArchitecture:")
        for key, value in info.items():
            print(f"  {key:20s}: {value}")
        
        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])
        
        # Training
        print(f"\nTraining for {EPOCHS} epochs...")
        best_loss = float('inf')
        
        for epoch in range(EPOCHS):
            loss = train_model(model, train_loader, optimizer, criterion, 
                             DEVICE, model_name)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")
            
            if loss < best_loss:
                best_loss = loss
                # Save best model
                torch.save(model.state_dict(), 
                          f"{model_name.replace(' ', '_').lower()}_best.pth")
        
        print(f"✓ Training complete! Best loss: {best_loss:.4f}")
        
        # Generate names
        print(f"\nGenerating 1000 names...")
        generated_names = generate_names(model, char_to_idx, idx_to_char, 
                                        DEVICE, num_names=1000)
        
        # Save generated names
        output_file = f"generated_{model_name.replace(' ', '_').lower()}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for name in generated_names:
                f.write(name + '\n')
        print(f"✓ Saved to: {output_file}")
        
        # Evaluate
        metrics = evaluate_model(generated_names, names)
        print_evaluation(metrics, model_name)
        
        # Store results
        results[model_name] = {
            'architecture': info,
            'metrics': metrics,
            'sample_names': generated_names[:20]
        }
    
    # Save results
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING AND EVALUATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  • vanilla_rnn_best.pth - Trained Vanilla RNN")
    print("  • bidirectional_lstm_best.pth - Trained BLSTM")
    print("  • rnn_with_attention_best.pth - Trained Attention RNN")
    print("  • generated_vanilla_rnn.txt - Generated names (Vanilla RNN)")
    print("  • generated_bidirectional_lstm.txt - Generated names (BLSTM)")
    print("  • generated_rnn_with_attention.txt - Generated names (Attention RNN)")
    print("  • evaluation_results.json - Complete evaluation results")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
