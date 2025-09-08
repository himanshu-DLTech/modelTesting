import ollama
import io
import sys
import traceback
import re
import pandas as pd

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
# Tasks
# ---------------------------------
tasks = [
    {
        "name": "Data Cleaning",
        "prompt": """Write Python code using pandas to load 'transactions.csv', standardize the date column 
        into YYYY-MM-DD format, fill missing numerical values with the column mean, 
        and print missing values before and after cleaning.""",
        "expected": ["Missing values before cleaning", "Missing values after cleaning"]
    },
    {
        "name": "Statistical Analysis",
        "prompt": """Generate Python code that loads 'scores.csv', computes mean, median, variance, skewness, 
        kurtosis of scores, and performs a t-test comparing students who studied >10 hours vs. <10 hours.""",
        "expected": ["mean", "t-test", "p-value"]
    },
    {
        "name": "Data Visualization",
        "prompt": """Write Python code using matplotlib to load 'sales.csv' and: 
        (1) line chart of monthly sales trends, 
        (2) bar chart comparing sales by month, 
        (3) histogram of sales distribution.""",
        "expected": ["plt.plot", "plt.bar", "plt.hist"]
    },
    {
        "name": "Machine Learning",
        "prompt": """Generate Python code that loads 'churn.csv', trains a logistic regression model 
        to predict churn. Include train-test split, scaling, training, accuracy, ROC-AUC, and confusion matrix.""",
        "expected": ["LogisticRegression", "accuracy", "roc_auc_score", "confusion_matrix"]
    },
    {
        "name": "Optimization",
        "prompt": """Write Python code that loads 'cities.csv', solves a Traveling Salesman Problem 
        using nearest-neighbor heuristic, and plots the route with matplotlib.""",
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
# Main Loop with Logging
# ---------------------------------
results = []

for task in tasks:
    print("="*70)
    print(f"🔹 Task: {task['name']}")

    # Step 1: Ask Ollama for code
    response = ollama.chat(model="qwen2.5-coder:32b", messages=[{"role": "user", "content": task["prompt"]}])
    code = extract_code(response["message"]["content"])

    print("\nGenerated Code Preview:\n", code[:250], "...\n")

    # Step 2: Run generated code
    result = run_generated_code(code)

    # Step 3: Check expected outputs
    score = 0
    if result["success"]:
        out = result["stdout"].lower()
        for keyword in task["expected"]:
            if keyword.lower() in out or keyword.lower() in code.lower():
                score += 1
        status = "✅ Success"
    else:
        status = "❌ Failed"

    # Step 4: Collect results
    result_text = (
        f"Task: {task['name']}\n"
        f"Status: {status}\n"
        f"Score: {score}/{len(task['expected'])}\n"
        f"Stdout: {result['stdout']}\n"
        f"Error: {result['error']}\n"
        f"{'-'*60}\n"
    )
    print(result_text)
    results.append(result_text)

# Save results to file
with open("results.txt", "w") as f:
    f.writelines(results)

print("📂 Results saved to results.txt")
