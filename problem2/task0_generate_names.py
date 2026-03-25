"""
TASK-0: Generate 1000 Indian Names
==================================

This script generates 1000 Indian names for training the RNN models.
Since we don't have LLM API access, we'll use a curated list of real Indian names
and generate variations to create a diverse dataset.

Usage:
    python task0_generate_names.py
"""

import random
import os

# Curated list of real Indian names (male and female)
INDIAN_NAMES = [
    # Male names
    "Aarav", "Aditya", "Advait", "Agastya", "Ajay", "Akash", "Akhil", "Amit", 
    "Amrit", "Anand", "Anil", "Anirudh", "Anish", "Ankit", "Arjun", "Arnav",
    "Aryan", "Ashish", "Ashok", "Atul", "Ayaan", "Ayush", "Bharat", "Chirag",
    "Darsh", "Deepak", "Dev", "Dhruv", "Dinesh", "Divyansh", "Gaurav", "Harsh",
    "Hemant", "Ishaan", "Jay", "Kabir", "Karan", "Kartik", "Keshav", "Krishna",
    "Kunal", "Laksh", "Manoj", "Mohit", "Nakul", "Naveen", "Nikhil", "Nitin",
    "Piyush", "Pranav", "Prateek", "Prithvi", "Rahul", "Raj", "Rajat", "Ravi",
    "Rohan", "Rohit", "Sai", "Sameer", "Sanjay", "Sanket", "Shiv", "Shreyas",
    "Siddharth", "Sumit", "Suraj", "Suresh", "Tanay", "Tanmay", "Tejas", "Varun",
    "Vedant", "Vikas", "Vikram", "Vinay", "Vineet", "Vishal", "Vivek", "Yash",
    
    # Female names
    "Aadhya", "Aanya", "Aarya", "Aditi", "Aishwarya", "Akshara", "Alisha", "Amrita",
    "Ananya", "Anjali", "Ankita", "Anushka", "Aparna", "Archana", "Arpita", "Avni",
    "Bhavana", "Bhavya", "Diya", "Divya", "Gauri", "Gargi", "Ishita", "Janhvi",
    "Jaya", "Jyoti", "Kavya", "Khushi", "Kiara", "Kritika", "Lakshmi", "Lavanya",
    "Madhuri", "Mahima", "Mansi", "Megha", "Meera", "Naina", "Navya", "Neha",
    "Nidhi", "Nikita", "Nisha", "Pooja", "Prachi", "Pragya", "Prerna", "Priya",
    "Priyanka", "Radha", "Radhika", "Riya", "Roshni", "Saanvi", "Sakshi", "Sangeeta",
    "Sanjana", "Sara", "Sarika", "Shakti", "Shivani", "Shreya", "Simran", "Sneha",
    "Sonam", "Sonali", "Swara", "Swati", "Tanvi", "Tara", "Trisha", "Urvashi",
    "Vani", "Varsha", "Vidya", "Zara",
    
    # Traditional names
    "Abhishek", "Brijesh", "Chandan", "Dharmendra", "Gagan", "Gopal", "Harish",
    "Jagdish", "Kailash", "Lalita", "Madhav", "Narendra", "Omkar", "Pankaj",
    "Ramesh", "Sachin", "Tarun", "Umesh", "Vijay", "Yogesh",
    
    "Anuradha", "Bharti", "Chandni", "Deepika", "Geeta", "Hemlata", "Indira",
    "Jayanti", "Kamala", "Lalita", "Mandira", "Nandini", "Padma", "Renuka",
    "Savita", "Suman", "Sunita", "Urmila", "Vandana", "Yamini"
]

# Common prefixes and suffixes for Indian names
PREFIXES = ["Maha", "Sri", "Prem", "Raj", "Hari", "Sita", "Ram", "Dev", "Kumar", ""]
SUFFIXES = ["raj", "kumar", "lal", "deep", "jit", "esh", "ini", "ika", "vati", "devi", ""]

def generate_variations(base_names, target_count=1000):
    """
    Generate variations of names to reach target count
    
    Args:
        base_names: List of base names
        target_count: Target number of names
        
    Returns:
        List of names
    """
    names = set(base_names)
    
    # Add original names
    all_names = list(names)
    
    # Generate variations by combining prefixes and suffixes
    attempts = 0
    max_attempts = target_count * 10
    
    while len(all_names) < target_count and attempts < max_attempts:
        attempts += 1
        
        # Method 1: Add prefix
        if random.random() < 0.3:
            base = random.choice(base_names)
            prefix = random.choice(PREFIXES)
            if prefix:
                new_name = prefix + base
                if new_name not in all_names and len(new_name) >= 4:
                    all_names.append(new_name)
        
        # Method 2: Add suffix
        elif random.random() < 0.3:
            base = random.choice(base_names)
            suffix = random.choice(SUFFIXES)
            if suffix:
                new_name = base + suffix
                if new_name not in all_names and len(new_name) >= 4:
                    all_names.append(new_name)
        
        # Method 3: Combine two names
        elif random.random() < 0.2:
            name1 = random.choice(base_names)
            name2 = random.choice(base_names)
            # Take first part of name1 and last part of name2
            split_point = len(name1) // 2
            new_name = name1[:split_point] + name2[split_point:]
            if new_name not in all_names and 4 <= len(new_name) <= 15:
                all_names.append(new_name)
        
        # Method 4: Just duplicate some originals if needed
        else:
            base = random.choice(base_names)
            if all_names.count(base) < 3:  # Allow some duplicates
                all_names.append(base)
    
    # If still not enough, repeat some names
    while len(all_names) < target_count:
        all_names.append(random.choice(base_names))
    
    # Trim to exact count and shuffle
    all_names = all_names[:target_count]
    random.shuffle(all_names)
    
    return all_names

def generate_training_names(output_file='TrainingNames.txt', count=1000):
    """
    Generate training names and save to file
    
    Args:
        output_file: Output file path
        count: Number of names to generate
    """
    print("="*70)
    print("TASK-0: GENERATE INDIAN NAMES")
    print("="*70)
    print(f"Target: {count} names")
    print(f"Output: {output_file}")
    print("="*70 + "\n")
    
    # Generate names
    print("Generating names...")
    names = generate_variations(INDIAN_NAMES, count)
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for name in names:
            f.write(name + '\n')
    
    # Statistics
    unique_names = len(set(names))
    avg_length = sum(len(name) for name in names) / len(names)
    min_length = min(len(name) for name in names)
    max_length = max(len(name) for name in names)
    
    print(f"Generated {len(names)} names")
    print(f"Unique names: {unique_names}")
    print(f"Average length: {avg_length:.1f} characters")
    print(f"Length range: {min_length} - {max_length} characters")
    print(f"Saved to: {output_file}")
    
    # Show sample names
    print(f"\nSample names (first 20):")
    print("-" * 70)
    for i, name in enumerate(names[:20], 1):
        print(f"{i:2d}. {name}")
    
    print("\nTask-0 Complete!")
    
    return names

if __name__ == "__main__":
    names = generate_training_names('TrainingNames.txt', 1000)
