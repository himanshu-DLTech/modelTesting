import ollama
import io
import sys
import traceback
import re
import pandas as pd

# ---------------------------------
# CLI Argument: Model name
# ---------------------------------
if len(sys.argv) < 2:
    print("Usage: python llm_tester.py <model_name>")
    sys.exit(1)

MODEL_NAME = sys.argv[1]

# ---------------------------------
# Prepare Sample Input Data
# ---------------------------------

# 1. Transactions CSV
transactions_csv = """customer_id,date,amount,region
101,2022/07/15,250,North
102,07-16-2022,,East
103,2022-07-17,180,West
104,,220,North
105,2022/07/19,,South
106,2022/07/20,310,East
107,07-21-2022,NaN,West
108,2022/07/22,150,South
109,,200,East
110,2022-07-24,NaN,North
"""
with open("transactions.csv", "w") as f:
    f.write(transactions_csv)

# 2. Student scores dataset
df_scores = pd.DataFrame({
    "student_id": range(1, 21),
    "hours_studied": [5, 8, 12, 3, 9, 15, 11, 2, 14, 6, 7, 13, 10, 1, 16, 4, 12, 9, 8, 14],
    "score": [65, 70, 88, 60, 72, 95, 85, 55, 92, 68, 74, 90, 80, 58, 96, 63, 87, 76, 71, 91]
})
df_scores.to_csv("scores.csv", index=False)

# 3. Sales data
df_sales = pd.DataFrame({
    "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
    "sales": [1200, 1350, 1600, 1450, 1700, 1800, 2100, 2000, 2200, 2500, 2400, 2600],
})
df_sales.to_csv("sales.csv", index=False)

# 4. Customer churn dataset
df_churn = pd.DataFrame({
    "age": [25, 45, 35, 50, 23, 40, 31, 29, 48, 60, 36, 52, 27, 39, 42, 33, 46, 55, 38, 30],
    "income": [40000, 80000, 60000, 90000, 35000, 70000, 58000, 45000, 85000, 95000,
               62000, 88000, 37000, 65000, 73000, 55000, 81000, 91000, 66000, 50000],
    "tenure": [12, 36, 24, 48, 6, 30, 18, 14, 40, 50, 20, 45, 8, 28, 33, 22, 38, 47, 26, 16],
    "churn": [0,1,0,1,0,0,0,0,1,1,0,1,0,0,1,0,1,1,0,0]
})
df_churn.to_csv("churn.csv", index=False)

# 5. Cities for TSP
df_cities = pd.DataFrame({
    "city": list(range(10)),
    "x": [10, 20, 30, 15, 25, 35, 5, 40, 22, 18],
    "y": [5, 15, 25, 30, 12, 28, 8, 35, 20, 27]
})
df_cities.to_csv("cities.csv", index=False)


# ---------------------------------
# Tasks (with schema context in prompts)
# ---------------------------------
tasks = [
    {
        "name": "Data Cleaning",
        "prompt": """Dataset: transactions.csv
Schema:
- customer_id (int)
- date (string, various formats, some missing)
- amount (float, may contain NaN)
- region (string)

Task: Write Python code using pandas to load the dataset, 
standardize the 'date' column into YYYY-MM-DD format, 
fill missing numerical values with the column mean, 
and print missing values before and after cleaning.""",
        "expected": ["Missing values before cleaning", "Missing values after cleaning"]
    },
    {
        "name": "Statistical Analysis",
        "prompt": """Dataset: scores.csv
Schema:
- student_id (int)
- hours_studied (int)
- score (int)

Task: Generate Python code that computes mean, median, variance, skewness, kurtosis 
of 'score', and performs a t-test comparing students who studied >10 hours vs. <10 hours.""",
        "expected": ["mean", "t-test", "p-value"]
    },
    {
        "name": "Data Visualization",
        "prompt": """Dataset: sales.csv
Schema:
- month (string, Jan–Dec)
- sales (int)

Task: Write Python code using matplotlib to:
1. Plot a line chart of monthly sales trends.
2. Plot a bar chart comparing sales by month.
3. Plot a histogram of sales distribution.""",
        "expected": ["plt.plot", "plt.bar", "plt.hist"]
    },
    {
        "name": "Machine Learning",
        "prompt": """Dataset: churn.csv
Schema:
- age (int)
- income (int)
- tenure (int, months)
- churn (0 = no, 1 = yes)

Task: Generate Python code that trains a logistic regression model 
to predict churn. Include train-test split, scaling, model training, 
accuracy score, ROC-AUC score, and a confusion matrix plot.""",
        "expected": ["LogisticRegression", "accuracy", "roc_auc_score", "confusion_matrix"]
    },
    {
        "name": "Optimization",
        "prompt": """Dataset: cities.csv
Schema:
- city (int)
- x (int, x-coordinate)
- y (int, y-coordinate)

Task: Write Python code that simulates the Traveling Salesman Problem (TSP) 
for these 10 cities using the nearest-neighbor heuristic 
and plots the resulting route using matplotlib.""",
        "expected": ["nearest", "plt.plot"]
    }
]


# ---------------------------------
# Helpers
# ---------------------------------
def extract_code(text: str):
    """Extract Python code from markdown-style response."""
    match = re.findall(r"```python(.*?)```", text, re.S)
    if match:
        return match[0]
    return text

def run_generated_code(code: str):
    """Run LLM-generated code and capture stdout/errors."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    result = {"success": True, "stdout": "", "error": ""}
    try:
        exec(code, {})
    except Exception:
        result["success"] = False
        result["error"] = traceback.format_exc()
    result["stdout"] = sys.stdout.getvalue()
    sys.stdout = old_stdout
    return result


# ---------------------------------
# Main Loop (Summary Only)
# ---------------------------------
total_score = 0
max_score = 0
passed_cases = 0

for task in tasks:
    print(f"Running task: {task['name']}...")

    # Step 1: Ask Ollama for code
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": task["prompt"]}])
    code = extract_code(response["message"]["content"])

    # Step 2: Run generated code
    result = run_generated_code(code)

    # Step 3: Check expected outputs
    score = 0
    if result["success"]:
        out = result["stdout"].lower()
        for keyword in task["expected"]:
            if keyword.lower() in out or keyword.lower() in code.lower():
                score += 1
        if score == len(task["expected"]):
            passed_cases += 1

    total_score += score
    max_score += len(task["expected"])

# ---------------------------------
# Final Summary
# ---------------------------------
summary = (
    f"=== FINAL SUMMARY ===\n"
    f"Model: {MODEL_NAME}\n"
    f"Total Score: {total_score}/{max_score}\n"
    f"Passed Cases: {passed_cases}/{len(tasks)}\n"
)

print(summary)

# Save only summary to file
with open("results.txt", "w") as f:
    f.write(summary)

print("📂 Results saved to results.txt")
