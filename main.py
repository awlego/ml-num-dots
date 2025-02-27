import ollama
import pandas as pd
import os
from datetime import datetime
import time
from tqdm import tqdm
import itertools
from itertools import product

# Dictionary to store results by model
results_by_model = {}
start_time = time.time()
total_tokens = 0

# eventually test anthropic / openai too
models_to_test = ['deepseek-r1:1.5b', 'deepseek-r1:7b', 'deepseek-r1:14b', 'deepseek-r1:32b']
eval_model = 'llama3'

prompts_templates = [
    "Do not use code. How many periods/dots '.' are in the following sequence?  Sequence: {}",
    "Do not use code. How many periods/dots '.' are in the following sequence?  Repeat your reasoning to double check your answer. If they match, output the number, otherwise try again. Sequence: {}",
]
# Enable/disable specific prompt templates (by index)
enabled_prompts = [0]  # Only use the first prompt template for this experiment

sequences = [
    lambda n: n * '.',      # Just dots
    lambda n: n * '. '      # Dots with spaces
]
# Enable/disable specific sequences (by index)
enabled_sequences = [1]

num_repeats = 10

max_dots_in_sequence = 25
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# First pass: collect responses model by model
for model in models_to_test:
    print(f"\n\n===== Starting tests for model: {model} =====\n")
    model_results = []
    
    # Run only enabled combinations for this model
    for prompt_idx in enabled_prompts:
        for seq_idx in enabled_sequences:
            prompt_template = prompts_templates[prompt_idx]
            sequence_generator = sequences[seq_idx]
            
            combo_idx = prompt_idx * len(sequences) + seq_idx + 1
            for num_dots in tqdm(range(1, max_dots_in_sequence + 1), 
                                desc=f"Model: {model}, Prompt: {prompt_idx+1}, Sequence: {seq_idx+1}"):
                sequence = sequence_generator(num_dots)
                prompt = prompt_template.format(sequence)
                
                # Repeat each prompt num_repeats times
                for repeat_idx in range(num_repeats):
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
                    model_results.append({
                        "model": model,
                        "prompt_type": prompt_idx + 1,
                        "sequence_type": seq_idx + 1,
                        "repeat_num": repeat_idx + 1,  # Add repeat number to track which repetition this is
                        "prompt": prompt,
                        "actual_count": num_dots,
                        "model_answer": response['message']['content'],
                        "extracted_number": None,
                        "correct": False
                    })
                
                # Print progress metrics (only once per num_dots, not for each repeat)
                if num_dots % 10 == 0:
                    elapsed_time = time.time() - start_time
                    tests_per_second = len(model_results) / elapsed_time
                    tokens_per_second = total_tokens / elapsed_time if total_tokens > 0 else 0
                    
                    print(f"\nProgress Update - Model: {model}, Prompt: {prompt_idx+1}, Sequence: {seq_idx+1}, Test {num_dots}/{max_dots_in_sequence}:")
                    print(f"Last response time: {iteration_time:.2f}s")
                    print(f"Average tests/second: {tests_per_second:.2f}")
                    if total_tokens > 0:
                        print(f"Average tokens/second: {tokens_per_second:.2f}")
                        print(f"Total tokens so far: {total_tokens}")
    
    # Second pass: evaluate all responses for this model
    print(f"\nEvaluating responses for model: {model}")
    for result in tqdm(model_results, desc=f"Extracting final answers for {model}..."):
        eval_prompt = f"Extract ONLY the final answer number from this response. Output just the number, nothing else: {result['model_answer']}"
        eval_response = ollama.chat(
            model=eval_model,
            messages=[{'role': 'user', 'content': eval_prompt}]
        )
        try:
            extracted_number = int(eval_response['message']['content'].strip())
            result['extracted_number'] = extracted_number
            result['correct'] = extracted_number == result['actual_count']
        except ValueError:
            pass
    
    # Save results for this model immediately
    model_df = pd.DataFrame(model_results)
    
    # Basic analysis for this model
    accuracy = model_df["correct"].mean()
    print(f"\nModel {model} accuracy: {accuracy:.2%}")
    
    # Save model results
    model_filename = model.replace(':', '_')
    output_file = os.path.join(output_dir, f"dot_counting_results_{model_filename}_{timestamp}.csv")
    model_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Also save as pickle
    pickle_file = os.path.join(output_dir, f"dot_counting_results_{model_filename}_{timestamp}.pkl")
    model_df.to_pickle(pickle_file)
    print(f"DataFrame saved as pickle to {pickle_file}")
    
    # Store results for combined analysis later
    results_by_model[model] = model_results

# Combine all results for final analysis
all_results = list(itertools.chain.from_iterable(results_by_model.values()))
df = pd.DataFrame(all_results)

# Overall analysis
overall_accuracy = df["correct"].mean()
print(f"\nOverall accuracy across all models: {overall_accuracy:.2%}")

# Find where models fail consistently
if not all(df["correct"]):
    failure_points = df[~df["correct"]]["actual_count"].value_counts().sort_index()
    print(f"\nFailure frequency by dot count:")
    print(failure_points)

# Save the complete results
complete_output_file = os.path.join(output_dir, f"dot_counting_results_ALL_MODELS_{timestamp}.csv")
df.to_csv(complete_output_file, index=False)
complete_pickle_file = os.path.join(output_dir, f"dot_counting_results_ALL_MODELS_{timestamp}.pkl")
df.to_pickle(complete_pickle_file)
print(f"Complete results saved to {complete_output_file} and {complete_pickle_file}")