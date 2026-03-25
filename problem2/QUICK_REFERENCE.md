# Quick Reference Guide
## RNN Name Generation Project

---

## ⚡ Quick Commands

### Run Everything (Complete Pipeline)
```bash
python run_all.py
```

### Run Individual Tasks
```bash
# Task 0: Generate training data
python task0_generate_names.py

# Task 1: Test models
python task1_models.py

# Train all models
python train_and_evaluate.py

# Evaluate and report
python evaluate_and_report.py
```

### Run Specific Task Only
```bash
python run_all.py --task 0  # Generate data
python run_all.py --task 1  # Test models
python run_all.py --task 2  # Train models
python run_all.py --task 3  # Evaluate
```

---

## 📋 File Checklist

### Input Files (Generated)
- [ ] `TrainingNames.txt` - 1000 training names

### Python Scripts (Provided)
- [ ] `task0_generate_names.py` - Data generation
- [ ] `task1_models.py` - Model implementations
- [ ] `train_and_evaluate.py` - Training pipeline
- [ ] `evaluate_and_report.py` - Evaluation
- [ ] `run_all.py` - Master script

### Output Files (After Training)
- [ ] `vanilla_rnn_best.pth` - Trained Vanilla RNN
- [ ] `bidirectional_lstm_best.pth` - Trained BLSTM
- [ ] `rnn_with_attention_best.pth` - Trained Attention RNN
- [ ] `generated_vanilla_rnn.txt` - Generated names (Vanilla)
- [ ] `generated_bidirectional_lstm.txt` - Generated names (BLSTM)
- [ ] `generated_rnn_with_attention.txt` - Generated names (Attention)
- [ ] `evaluation_results.json` - Quantitative metrics
- [ ] `detailed_evaluation.json` - Detailed analysis

---

## 🎯 Model Architectures Summary

### 1. Vanilla RNN
```
Parameters: ~137K
Layers: Embedding → RNN (2 layers) → Linear
Equation: h_t = tanh(W_ih·x_t + W_hh·h_{t-1} + b)
```

### 2. Bidirectional LSTM
```
Parameters: ~268K
Layers: Embedding → BiLSTM (2 layers) → Linear
Gates: Input, Forget, Cell, Output
Bidirectional: Processes forward and backward
```

### 3. RNN with Attention
```
Parameters: ~182K
Layers: Embedding → GRU → Attention → Linear
Attention: Additive (Bahdanau) style
Context: Weighted sum of hidden states
```

---

## 📊 Evaluation Metrics

### Novelty Rate
```
novelty_rate = (names not in training / total generated) × 100%
Target: > 80%
```

### Diversity
```
diversity = (unique names / total generated) × 100%
Target: > 70%
```

### Realism
```
Assessed by:
- Length (3-12 chars)
- No excessive repetition
- Balanced vowels/consonants
- Phonetically plausible
```

---

## 🔧 Common Modifications

### Change Training Duration
```python
# In train_and_evaluate.py, line ~32
EPOCHS = 50  # Increase for better quality
```

### Adjust Model Size
```python
# In train_and_evaluate.py, lines 24-27
EMBEDDING_DIM = 64   # Embedding size
HIDDEN_SIZE = 128    # Hidden layer size
NUM_LAYERS = 2       # Number of layers
```

### Generate More Names
```python
# In train_and_evaluate.py, line ~180
generate_names(..., num_names=1000)  # Change number
```

### Control Randomness
```python
# In train_and_evaluate.py, line ~141
temperature=0.8  # 0.5=conservative, 1.5=creative
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Fix:** Install PyTorch
```bash
pip install torch numpy tqdm
```

### Issue: "CUDA out of memory"
**Fix:** Reduce batch size
```python
BATCH_SIZE = 32  # or 16
```

### Issue: Training too slow
**Fix:** 
1. Reduce epochs: `EPOCHS = 20`
2. Use GPU if available
3. Reduce model size

### Issue: Poor name quality
**Fix:**
1. Train longer: `EPOCHS = 100`
2. Increase model: `HIDDEN_SIZE = 256`
3. Adjust temperature: `temperature=0.6`

### Issue: Names too repetitive
**Fix:**
1. Increase temperature: `temperature=1.0`
2. Train longer
3. Try different model (Attention RNN)

---

## 📈 Expected Training Time

| Hardware | Time per Epoch | Total Time (50 epochs) |
|----------|---------------|----------------------|
| CPU (4 cores) | ~30-40 sec | 25-35 minutes |
| GPU (mid-range) | ~5-10 sec | 5-10 minutes |
| GPU (high-end) | ~2-5 sec | 2-5 minutes |

---

## 📝 Typical Results

### Vanilla RNN
```
Novelty: 75-85%
Diversity: 60-70%
Sample: Aarav, Aditi, Rohan, Priya, Karan
```

### Bidirectional LSTM
```
Novelty: 80-90%
Diversity: 70-80%
Sample: Aditya, Anushka, Siddharth, Lavanya
```

### RNN with Attention
```
Novelty: 85-95%
Diversity: 75-85%
Sample: Advait, Aadhya, Vihaan, Saanvi
```

---

## 🎓 Key Concepts

### Character-Level Modeling
- Each character is a token
- Model learns character sequences
- Can generate novel combinations

### Teacher Forcing
- During training: use true previous character
- During generation: use model's prediction

### Sampling Strategies
- **Greedy:** Always pick most likely
- **Temperature:** Control randomness
- **Top-k:** Sample from k most likely

### Attention Mechanism
- Focus on relevant parts of input
- Learn what to attend to
- Improves long-range dependencies

---

## 📚 Further Reading

### Papers
- Hochreiter & Schmidhuber (1997) - LSTM
- Bahdanau et al. (2014) - Attention
- Karpathy (2015) - Char-RNN blog

### Code Examples
- PyTorch RNN Tutorial
- Karpathy's char-rnn
- Attention Is All You Need

---

## ✅ Deliverables Checklist

For submission, ensure you have:

- [ ] Source code for all models ✓
  - task1_models.py
  
- [ ] Generated name samples ✓
  - generated_vanilla_rnn.txt
  - generated_bidirectional_lstm.txt
  - generated_rnn_with_attention.txt
  
- [ ] Evaluation scripts ✓
  - evaluate_and_report.py
  
- [ ] Report ✓
  - evaluation_results.json
  - detailed_evaluation.json
  - README_RNN_PROJECT.md

- [ ] Training data ✓
  - TrainingNames.txt

---

## 🚀 Pro Tips

1. **Start small:** Test with EPOCHS=5 first
2. **Monitor loss:** Should decrease over time
3. **Save checkpoints:** Models save automatically
4. **Experiment:** Try different hyperparameters
5. **Visualize:** Check generated names regularly
6. **Compare:** Run all models to see differences
7. **Document:** Keep notes on what works

---

**Good luck! 🎉**
