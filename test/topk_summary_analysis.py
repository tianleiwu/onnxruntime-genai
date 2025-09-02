import pandas as pd
import numpy as np

def analyze_and_evaluate_rules():
    """
    Loads the benchmark CSV, derives heuristic rules, and evaluates their accuracy.
    """
    try:
        df = pd.read_csv('topk_4090_comprehensive_summary.csv')
    except FileNotFoundError:
        print("Error: 'topk_4090_comprehensive_summary.csv' not found.")
        print("Please make sure the benchmark data file is in the same directory.")
        return

    # --- 1. Data Preparation ---
    
    # Function to parse the 'best' column and extract the optimal sort_size.
    # Returns 0 for SELECTION_SORT.
    def parse_best_sort_size(best_str):
        if 'SELECTION_SORT' in best_str:
            return 0
        elif 'HYBRID' in best_str:
            try:
                return int(best_str.split('s=')[1].split(',')[0])
            except (IndexError, ValueError):
                return -1  # Should not happen with clean data
        return -1

    df['best_sort_size'] = df['best'].apply(parse_best_sort_size)
    
    # --- 2. Rule-Based Classification Function ---

    def predict_sort_size(row):
        v, b, k = row['vocab_size'], row['batch_size'], row['k']

        """
        # Rule 1: For k=1, Selection Sort is consistently the fastest.
        if k < 8:
            return 0

        # Rule 2: For very large vocabularies, sort_size=4096 is the clear winner.
        if v >= 180224:
            return 4096

        # Rule 3: For medium-to-large vocabularies, 2048 is the most common optimal size.
        if b >= 4 and v >= 49152:
            return 2048
        if v >= 65536:
            return 2048

        # Rule 4: For smaller vocabularies with smaller k, Selection Sort is faster.
        if k < 32 and v < 49152:
            return 0

        # Rule 5: Default for remaining cases (e.g., small vocab with large k).
        return 1024
        """
        if k <= 4:
            return 0
        
        # Rule 2: For very large vocabularies, sort_size=4096 is the clear winner.
        if v >= 147456:
            return 4096
        elif v < 49152:
            # Rule 4: For smaller vocabularies with smaller k, Selection Sort is faster.
            if k <= 8:
                return 0
        else:
            if k < 8:
                return 0
            # For medium-to-large vocabularies, 2048 is the most common optimal size.
            if b >= 4 and v >= 49152:
                return 2048

            if v >= 65536:
                return 2048

        # Rule 5: Default for remaining cases (e.g., small vocab with large k).
        return 1024

    # --- 3. Evaluation ---

    # Apply the prediction function to the entire dataset
    df['predicted_sort_size'] = df.apply(predict_sort_size, axis=1)

    # Calculate the accuracy
    accuracy = (df['best_sort_size'] == df['predicted_sort_size']).mean()
    
    print(f"Analysis of 'topk_4090_comprehensive_summary.csv'")
    print("-" * 50)
    print(f"Total number of benchmark configurations: {len(df)}")
    print(f"Accuracy of the derived rule-based function: {accuracy * 100:.2f}%")
    print("-" * 50)
    
    # Show a breakdown of misclassifications to understand where the model fails
    mismatches = df[df['best_sort_size'] != df['predicted_sort_size']]
    print(f"\nFound {len(mismatches)} misclassifications.")
    if not mismatches.empty:
        print("Breakdown of top 5 misclassifications (Actual vs. Predicted):")
        print(mismatches[['best_sort_size', 'predicted_sort_size']].value_counts().nlargest(5))


if __name__ == '__main__':
    analyze_and_evaluate_rules()
