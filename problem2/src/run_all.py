"""
Master Script - Run Complete Pipeline
======================================

This script runs the entire RNN name generation pipeline:
1. Generate training data (Task 0)
2. Test model implementations (Task 1)
3. Train all models and generate names (Task 1 & 2)
4. Evaluate and create report (Task 2 & 3)

Usage:
    python run_all.py
    
    or for individual tasks:
    python run_all.py --task 0
    python run_all.py --task 1
    python run_all.py --task 2
    python run_all.py --task 3
"""

import sys
import os
import subprocess
import argparse


def run_task0():
    """Run Task 0: Generate training names"""
    print("\n" + "="*70)
    print("TASK 0: GENERATING TRAINING DATA")
    print("="*70 + "\n")
    
    result = subprocess.run([sys.executable, 'task0_generate_names.py'], 
                          capture_output=False)
    
    if result.returncode == 0:
        print("\nTask 0 Complete!")
        return True
    else:
        print("\nTask 0 Failed!")
        return False


def run_task1():
    """Run Task 1: Test model implementations"""
    print("\n" + "="*70)
    print("TASK 1: TESTING MODEL IMPLEMENTATIONS")
    print("="*70 + "\n")
    
    result = subprocess.run([sys.executable, 'task1_models.py'], 
                          capture_output=False)
    
    if result.returncode == 0:
        print("\nTask 1 Complete!")
        return True
    else:
        print("\nTask 1 Failed!")
        return False


def run_training():
    """Run Training: Train all models and generate names"""
    print("\n" + "="*70)
    print("TRAINING ALL MODELS")
    print("="*70 + "\n")
    
    print("⚠️  This may take 15-30 minutes on CPU or 5-10 minutes on GPU")
    print("   Press Ctrl+C to cancel\n")
    
    result = subprocess.run([sys.executable, 'train_and_evaluate.py'], 
                          capture_output=False)
    
    if result.returncode == 0:
        print("\nTraining Complete!")
        return True
    else:
        print("\nTraining Failed!")
        return False


def run_evaluation():
    """Run Task 2 & 3: Evaluation and reporting"""
    print("\n" + "="*70)
    print("TASK 2 & 3: EVALUATION AND REPORTING")
    print("="*70 + "\n")
    
    result = subprocess.run([sys.executable, 'evaluate_and_report.py'], 
                          capture_output=False)
    
    if result.returncode == 0:
        print("\nEvaluation Complete!")
        return True
    else:
        print("\nEvaluation Failed!")
        return False


def check_files():
    """Check if required files exist"""
    required_files = [
        'task0_generate_names.py',
        'task1_models.py',
        'train_and_evaluate.py',
        'evaluate_and_report.py'
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print("Missing required files:")
        for f in missing:
            print(f"   - {f}")
        return False
    
    return True


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Run RNN Name Generation Pipeline')
    parser.add_argument('--task', type=int, choices=[0, 1, 2, 3], 
                       help='Run specific task (0: generate data, 1: test models, 2: train, 3: evaluate)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (use existing models)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("RNN NAME GENERATION - MASTER PIPELINE")
    print("="*70)
    
    # Check files
    if not check_files():
        print("\nPlease ensure all required files are in the current directory")
        return
    
    # Run specific task
    if args.task is not None:
        if args.task == 0:
            run_task0()
        elif args.task == 1:
            run_task1()
        elif args.task == 2:
            run_training()
        elif args.task == 3:
            run_evaluation()
        return
    
    # Run complete pipeline
    print("\nRunning complete pipeline...")
    print("\nSteps:")
    print("  1. Generate training data (Task 0)")
    print("  2. Test model implementations (Task 1)")
    print("  3. Train all models (Task 1 & 2)")
    print("  4. Evaluate and report (Task 2 & 3)")
    
    response = input("\nProceed? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Cancelled.")
        return
    
    # Task 0
    if not os.path.exists('TrainingNames.txt'):
        if not run_task0():
            return
    else:
        print("\nTrainingNames.txt already exists, skipping Task 0")
    
    # Task 1
    if not run_task1():
        return
    
    # Training
    if not args.skip_training:
        if not run_training():
            return
    else:
        print("\nSkipping training (using existing models)")
    
    # Evaluation
    if not run_evaluation():
        return
    
    print("Complete")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nstarted.....")
        sys.exit(1)
