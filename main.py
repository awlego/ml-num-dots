import ollama
import pandas as pd
import os
from datetime import datetime

results = []
# Test counts from 1 to 100 dots
for num_dots in range(1, 101):
    prompt = f"How many periods/dots (.) are in this sequence? Do not use code. {num_dots * '.'}"
    
    # Call the model
    response = ollama.chat(
        model='deepseek-r1:7b',
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )
    
    answer = response['message']['content']
    
    # Record results
    results.append({
        "actual_count": num_dots,
        "model_answer": answer,
        "correct": str(num_dots) in answer  # Simple check
    })
    
    # Optional: Print progress 
    if num_dots % 10 == 0:
        print(f"Completed {num_dots} tests")

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
output_file = os.path.join(output_dir, f"dot_counting_results_{timestamp}.csv")
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")