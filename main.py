import ollama
import pandas as pd
import os
from datetime import datetime
import time
from tqdm import tqdm
import itertools
from itertools import product

results = []
start_time = time.time()
total_tokens = 0

# eventually test anthropic / openai too
models_to_test = ['deepseek-r1:1.5b', 'deepseek-r1:7b', 'deepseek-r1:14b', 'deepseek-r1:32b']
eval_model = 'llama3'


prompts_templates = [
    "Do not use code. How many periods/dots '.' are in the following sequence?  Sequence: {}",
    "Do not use code. How many periods/dots '.' are in the following sequence?  Repeat your reasoning to double check your answer. If they match, output the number, otherwise try again. Sequence: {}",
]

sequences = [
    lambda n: n * '.',      # Just dots
    lambda n: n * '. '      # Dots with spaces
]

max_dots_in_sequence = 100

# First pass: collect all responses
for model, (prompt_idx, prompt_template), (seq_idx, sequence_generator) in product(
    models_to_test,
    enumerate(prompts_templates),
    enumerate(sequences)
):
    combo_idx = prompt_idx * len(sequences) + seq_idx + 1
    for num_dots in tqdm(range(1, max_dots_in_sequence + 1), 
                        desc=f"Model: {model}, Prompt: {prompt_idx+1}, Sequence: {seq_idx+1}"):
        sequence = sequence_generator(num_dots)
        prompt = prompt_template.format(sequence)
        
        iteration_start = time.time()
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        iteration_time = time.time() - iteration_start
        
        # Track tokens
        if 'prompt_eval_count' in response:
            total_tokens += response['prompt_eval_count']
        if 'eval_count' in response:
            total_tokens += response['eval_count']
        
        # Record results without evaluation
        results.append({
            "model": model,
            "prompt_type": prompt_idx + 1,
            "sequence_type": seq_idx + 1,
            "prompt": prompt,
            "actual_count": num_dots,
            "model_answer": response['message']['content'],
            "extracted_number": None,
            "correct": False
        })
        
        # Print progress metrics
        if num_dots % 10 == 0:
            elapsed_time = time.time() - start_time
            tests_per_second = len(results) / elapsed_time
            tokens_per_second = total_tokens / elapsed_time if total_tokens > 0 else 0
            
            print(f"\nProgress Update - Model: {model}, Prompt: {prompt_idx+1}, Sequence: {seq_idx+1}, Test {num_dots}/100:")
            print(f"Last response time: {iteration_time:.2f}s")
            print(f"Average tests/second: {tests_per_second:.2f}")
            if total_tokens > 0:
                print(f"Average tokens/second: {tokens_per_second:.2f}")
                print(f"Total tokens so far: {total_tokens}")

# Second pass: evaluate all responses at once
print("\nEvaluating responses")
for result in tqdm(results, desc="Extracting final answers from responses..."):
    eval_prompt = f"Extract ONLY the final answer number from this response. Output just the number, nothing else: {result['model_answer']}"
    eval_response = ollama.chat(
        model=eval_model,
        messages=[{'role': 'user', 'content': eval_prompt}]
    )
    print(eval_response['message']['content'].strip())
    try:
        extracted_number = int(eval_response['message']['content'].strip())
        result['extracted_number'] = extracted_number
        result['correct'] = extracted_number == result['actual_count']
    except ValueError:
        pass

# Create and analyze results dataframe
df = pd.DataFrame(results)

# Basic analysis
accuracy = df["correct"].mean()
print(f"Overall accuracy: {accuracy:.2%}")

# Find where model fails consistently
if not all(df["correct"]):
    failure_points = df[~df["correct"]]["actual_count"].tolist()
    print(f"Model failed at these counts: {failure_points}")

# Save results to CSV file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Group results by model and save separate files for each model
for model in models_to_test:
    model_df = df[df["model"] == model]
    
    # Skip if no results for this model
    if len(model_df) == 0:
        continue
        
    model_filename = model.replace(':', '_')
    output_file = os.path.join(output_dir, f"dot_counting_results_{model_filename}_{timestamp}.csv")
    model_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Save the complete results as well
complete_pickle_file = os.path.join(output_dir, f"dot_counting_results_ALL_MODELS_{timestamp}.pkl")
df.to_pickle(complete_pickle_file)
print(f"Complete results saved to {complete_pickle_file}")