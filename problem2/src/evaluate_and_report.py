"""
TASK-2 & 3: Evaluation and Analysis
====================================

This script provides:
- Quantitative evaluation (novelty, diversity)
- Qualitative analysis (realism, failure modes)
- Comparison across models
- Report generation

Usage:
    python evaluate_and_report.py
"""

import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from typing import List, Dict
import os


def load_generated_names(filename):
    """Load generated names from file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_training_names(filename='TrainingNames.txt'):
    """Load training names"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def compute_novelty_rate(generated_names, training_names):
    """
    Compute novelty rate: percentage of generated names not in training set
    
    Args:
        generated_names: List of generated names
        training_names: List of training names
        
    Returns:
        novelty_rate: Percentage (0-100)
        novel_names: List of novel names
    """
    training_set = set(training_names)
    novel_names = [name for name in generated_names if name not in training_set]
    novelty_rate = (len(novel_names) / len(generated_names)) * 100
    
    return novelty_rate, novel_names


def compute_diversity(generated_names):
    """
    Compute diversity: number of unique names / total names
    
    Args:
        generated_names: List of generated names
        
    Returns:
        diversity: Percentage (0-100)
        unique_count: Number of unique names
    """
    unique_names = set(generated_names)
    diversity = (len(unique_names) / len(generated_names)) * 100
    
    return diversity, len(unique_names)


def analyze_name_structure(names):
    """
    Analyze structural properties of names
    
    Returns:
        Dictionary with structural statistics
    """
    lengths = [len(name) for name in names]
    
    # Character frequency
    all_chars = ''.join(names)
    char_freq = Counter(all_chars)
    
    # First letter distribution
    first_letters = Counter([name[0] for name in names if name])
    
    # Vowel/consonant ratio
    vowels = set('aeiouAEIOU')
    vowel_counts = [sum(1 for c in name if c in vowels) for name in names]
    consonant_counts = [len(name) - v for name, v in zip(names, vowel_counts)]
    
    return {
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'avg_vowels': np.mean(vowel_counts),
        'avg_consonants': np.mean(consonant_counts),
        'most_common_chars': char_freq.most_common(10),
        'most_common_first_letters': first_letters.most_common(10)
    }


def assess_realism(names, training_names):
    """
    Assess realism of generated names
    
    Returns:
        Dictionary with realism metrics
    """
    # Check for realistic patterns
    realistic_count = 0
    issues = {
        'too_short': 0,
        'too_long': 0,
        'repeated_chars': 0,
        'unusual_patterns': 0
    }
    
    for name in names:
        is_realistic = True
        
        # Length check
        if len(name) < 3:
            issues['too_short'] += 1
            is_realistic = False
        elif len(name) > 12:
            issues['too_long'] += 1
            is_realistic = False
        
        # Repeated characters check (more than 2 in a row)
        for i in range(len(name) - 2):
            if name[i] == name[i+1] == name[i+2]:
                issues['repeated_chars'] += 1
                is_realistic = False
                break
        
        # Check for consonant clusters (more than 4 consonants)
        vowels = set('aeiouAEIOU')
        consonant_run = 0
        for char in name:
            if char not in vowels:
                consonant_run += 1
                if consonant_run > 4:
                    issues['unusual_patterns'] += 1
                    is_realistic = False
                    break
            else:
                consonant_run = 0
        
        if is_realistic:
            realistic_count += 1
    
    realism_rate = (realistic_count / len(names)) * 100
    
    return {
        'realism_rate': realism_rate,
        'realistic_count': realistic_count,
        'issues': issues
    }


def identify_failure_modes(names):
    """
    Identify common failure modes
    
    Returns:
        Dictionary with failure mode examples
    """
    failure_modes = {
        'repeated_characters': [],
        'too_short': [],
        'too_long': [],
        'consonant_heavy': [],
        'vowel_heavy': [],
        'unusual_combinations': []
    }
    
    vowels = set('aeiouAEIOU')
    
    for name in names:
        # Repeated characters
        for i in range(len(name) - 2):
            if name[i] == name[i+1] == name[i+2]:
                if len(failure_modes['repeated_characters']) < 5:
                    failure_modes['repeated_characters'].append(name)
                break
        
        # Too short
        if len(name) < 3:
            if len(failure_modes['too_short']) < 5:
                failure_modes['too_short'].append(name)
        
        # Too long
        if len(name) > 12:
            if len(failure_modes['too_long']) < 5:
                failure_modes['too_long'].append(name)
        
        # Consonant heavy
        consonant_ratio = sum(1 for c in name if c not in vowels) / len(name)
        if consonant_ratio > 0.8:
            if len(failure_modes['consonant_heavy']) < 5:
                failure_modes['consonant_heavy'].append(name)
        
        # Vowel heavy
        vowel_ratio = sum(1 for c in name if c in vowels) / len(name)
        if vowel_ratio > 0.7:
            if len(failure_modes['vowel_heavy']) < 5:
                failure_modes['vowel_heavy'].append(name)
    
    return failure_modes


def compare_models(model_results):
    """
    Compare performance across all models
    
    Args:
        model_results: Dictionary with results for each model
        
    Returns:
        Comparison dictionary
    """
    comparison = {}
    
    for model_name, results in model_results.items():
        comparison[model_name] = {
            'novelty_rate': results['novelty_rate'],
            'diversity': results['diversity'],
            'realism_rate': results['realism']['realism_rate'],
            'avg_length': results['structure']['avg_length']
        }
    
    return comparison


def generate_report(training_names, model_files):
    """
    Generate comprehensive evaluation report
    
    Args:
        training_names: List of training names
        model_files: Dictionary mapping model names to generated name files
    """
    print("="*70)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("="*70)
    
    all_results = {}
    
    for model_name, filename in model_files.items():
        print(f"\n{'#'*70}")
        print(f"EVALUATING: {model_name}")
        print(f"{'#'*70}\n")
        
        # Load generated names
        generated_names = load_generated_names(filename)
        print(f"✓ Loaded {len(generated_names)} generated names")
        
        # Novelty Rate
        novelty_rate, novel_names = compute_novelty_rate(generated_names, training_names)
        print(f"\nNOVELTY RATE: {novelty_rate:.2f}%")
        print(f"   Novel names: {len(novel_names)}/{len(generated_names)}")
        
        # Diversity
        diversity, unique_count = compute_diversity(generated_names)
        print(f"\nDIVERSITY: {diversity:.2f}%")
        print(f"   Unique names: {unique_count}/{len(generated_names)}")
        
        # Structural analysis
        structure = analyze_name_structure(generated_names)
        print(f"\nSTRUCTURAL ANALYSIS:")
        print(f"   Average length: {structure['avg_length']:.2f} ± {structure['std_length']:.2f}")
        print(f"   Length range: {structure['min_length']} - {structure['max_length']}")
        print(f"   Average vowels: {structure['avg_vowels']:.2f}")
        print(f"   Average consonants: {structure['avg_consonants']:.2f}")
        
        # Realism assessment
        realism = assess_realism(generated_names, training_names)
        print(f"\nREALISM ASSESSMENT:")
        print(f"   Realism rate: {realism['realism_rate']:.2f}%")
        print(f"   Realistic names: {realism['realistic_count']}/{len(generated_names)}")
        print(f"   Issues found:")
        for issue, count in realism['issues'].items():
            print(f"     - {issue}: {count}")
        
        # Failure modes
        failures = identify_failure_modes(generated_names)
        print(f"\nCOMMON FAILURE MODES:")
        for mode, examples in failures.items():
            if examples:
                print(f"   {mode}: {', '.join(examples[:3])}")
        
        # Sample names
        print(f"\nSAMPLE GENERATED NAMES (First 20):")
        for i, name in enumerate(generated_names[:20], 1):
            marker = "✓" if name not in training_names else "✗"
            print(f"   {i:2d}. {marker} {name}")
        
        # Store results
        all_results[model_name] = {
            'total_generated': len(generated_names),
            'novelty_rate': novelty_rate,
            'novel_count': len(novel_names),
            'diversity': diversity,
            'unique_count': unique_count,
            'structure': structure,
            'realism': realism,
            'failure_modes': failures,
            'samples': generated_names[:20]
        }
    
    # Model comparison
    print(f"\n{'='*70}")
    print("MODEL COMPARISON")
    print(f"{'='*70}\n")
    
    comparison = compare_models(all_results)
    
    # Print comparison table
    print(f"{'Model':30s} {'Novelty %':>12s} {'Diversity %':>12s} {'Realism %':>12s} {'Avg Length':>12s}")
    print("-" * 70)
    
    for model_name, metrics in comparison.items():
        print(f"{model_name:30s} "
              f"{metrics['novelty_rate']:>11.2f}% "
              f"{metrics['diversity']:>11.2f}% "
              f"{metrics['realism_rate']:>11.2f}% "
              f"{metrics['avg_length']:>11.2f}")
    
    # Determine best model for each metric
    print(f"\nBEST PERFORMERS:")
    
    best_novelty = max(comparison.items(), key=lambda x: x[1]['novelty_rate'])
    print(f"   Novelty: {best_novelty[0]} ({best_novelty[1]['novelty_rate']:.2f}%)")
    
    best_diversity = max(comparison.items(), key=lambda x: x[1]['diversity'])
    print(f"   Diversity: {best_diversity[0]} ({best_diversity[1]['diversity']:.2f}%)")
    
    best_realism = max(comparison.items(), key=lambda x: x[1]['realism_rate'])
    print(f"   Realism: {best_realism[0]} ({best_realism[1]['realism_rate']:.2f}%)")
    
    # Save detailed results
    with open('detailed_evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print("\n✓ Detailed results saved to: detailed_evaluation.json")
    
    return all_results


def main():
    """Main evaluation function"""
    # Load training names
    training_names = load_training_names('TrainingNames.txt')
    print(f"Loaded {len(training_names)} training names\n")
    
    # Model files
    model_files = {
        'Vanilla RNN': 'generated_vanilla_rnn.txt',
        'Bidirectional LSTM': 'generated_bidirectional_lstm.txt',
        'RNN with Attention': 'generated_rnn_with_attention.txt'
    }
    
    # Check if files exist
    for model_name, filename in model_files.items():
        if not os.path.exists(filename):
            print(f"⚠️  Warning: {filename} not found!")
            print(f"   Please run train_and_evaluate.py first.\n")
            return
    
    # Generate report
    results = generate_report(training_names, model_files)


if __name__ == "__main__":
    main()
