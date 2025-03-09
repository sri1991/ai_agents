Interview Questions and Expected Answers
1. Python Programming & Data Manipulation (Expert, 20)
Subtopics: Data structures, Memory optimization, Parallel processing
1.1 Beginner: Theoretical
Question: "What are the main Python data structures, and how do they differ in terms of use cases?"

Expected Answer:  
"The main data structures are lists (ordered, mutable), tuples (ordered, immutable), dictionaries (key-value pairs), and sets (unordered, unique elements).  

Use cases: Lists for sequences (e.g., student grades), tuples for fixed data (e.g., coordinates), dictionaries for mappings (e.g., user IDs), and sets for unique items (e.g., unique tags).  

Difference: Lists allow duplicates and indexing; tuples are faster for read-only data; dictionaries offer O(1) lookups; sets ensure uniqueness."

1.2 Intermediate: Practical
Question: "Write a Python function to merge two lists and remove duplicates, optimizing for memory usage. How would you test it?"

Expected Answer:  
Code:
python

def merge_unique_lists(list1, list2):
    return list(set(list1 + list2))  # Convert to set for uniqueness, back to list

# Test
list_a = [1, 2, 3, 2]
list_b = [3, 4, 5, 1]
result = merge_unique_lists(list_a, list_b)
print(result)  # Expected: [1, 2, 3, 4, 5]

Optimization: Uses set for O(n) deduplication instead of nested loops (O(n²)). Memory is minimized by avoiding intermediate lists.

Test: "I’d test with edge cases (empty lists, duplicates) and verify output with assert statements."

1.3 Challenging: Scenario
Question: "You need to process a 5GB dataset in Python with limited memory. Design a solution using parallel processing to compute the average of a numeric column. How do you handle memory constraints?"

Expected Answer:  
"I’d use pandas with chunksize and multiprocessing. Split the 5GB CSV into chunks (e.g., 100MB each), process in parallel, and aggregate results.  

Code:
python

from multiprocessing import Pool
import pandas as pd

def process_chunk(chunk):
    return chunk['numeric_column'].mean()

if __name__ == '__main__':
    chunk_size = 1000000
    means = []
    with Pool(processes=4) as pool:
        for chunk in pd.read_csv('data.csv', chunksize=chunk_size):
            means.append(pool.apply_async(process_chunk, (chunk,)))
    final_mean = sum(m.get() for m in means) / len(means)
    print(final_mean)

Memory Handling: Chunks reduce memory usage; multiprocessing leverages multiple cores. I’d monitor RAM with psutil and adjust chunksize if needed."

2. Machine Learning & Predictive Modeling (Expert, 20)
Subtopics: Supervised learning, Unsupervised learning, Feature engineering, Cross-validation
2.1 Beginner: Theoretical
Question: "What is the difference between supervised and unsupervised learning? Give an example of each."

Expected Answer:  
"Supervised learning uses labeled data to predict outcomes (e.g., Linear Regression for house price prediction). Unsupervised learning finds patterns in unlabeled data (e.g., K-Means for customer segmentation).  

Difference: Supervised requires targets; unsupervised explores structure."

2.2 Intermediate: Practical
Question: "Implement a simple feature engineering step to create a new feature from a dataset with ‘age’ and ‘income’ columns, then explain how it improves a model."

Expected Answer:  
Code:
python

import pandas as pd
data = pd.DataFrame({'age': [25, 30, 45], 'income': [50000, 60000, 80000]})
data['age_income_ratio'] = data['income'] / data['age']
print(data)

Explanation: "The age_income_ratio combines age and income, capturing earning potential per year. This can improve a model (e.g., credit risk) by providing a normalized feature, reducing multicollinearity."

2.3 Challenging: Scenario
Question: "You’re building a predictive model for customer churn with imbalanced data (5% churn). Design a solution using cross-validation and feature engineering to improve performance."

Expected Answer:  
"I’d use SMOTE for oversampling the minority class, engineer features like ‘tenure’ or ‘usage_frequency’, and apply stratified k-fold cross-validation.  

Code:
python

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd

data = pd.read_csv('churn.csv')
X = data.drop('churn', axis=1)
y = data['churn']
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)
skf = StratifiedKFold(n_splits=5)
model = RandomForestClassifier()
for train_idx, test_idx in skf.split(X_res, y_res):
    X_train, X_test = X_res[train_idx], X_res[test_idx]
    y_train, y_test = y_res[train_idx], y_res[test_idx]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Fold score: {score}")

Improvement: Stratification ensures class balance, SMOTE addresses imbalance, and new features enhance predictive power."

3. SQL & Database Querying (Expert, 20)
Subtopics: Indexing, Query optimization, NoSQL vs SQL trade-offs
3.1 Beginner: Theoretical
Question: "What is an index in SQL, and how does it improve query performance?"

Expected Answer:  
"An index is a data structure (e.g., B-tree) that speeds up data retrieval by providing a quick lookup for rows. It improves performance by reducing full table scans to indexed column searches, though it slows down writes due to maintenance."

3.2 Intermediate: Practical
Question: "Write a SQL query to find the total sales per region from tables ‘Sales’ and ‘Regions,’ with an index recommendation."

Expected Answer:  
Query:
sql

SELECT r.RegionName, SUM(s.SalesAmount) as TotalSales
FROM Sales s
INNER JOIN Regions r ON s.RegionID = r.RegionID
GROUP BY r.RegionName;

Index: "I’d recommend an index on Sales.RegionID and Regions.RegionID to speed up the JOIN operation."

3.3 Challenging: Scenario
Question: "A query on a 1TB database is taking 10 minutes. Suggest a strategy to optimize it, considering indexing and NoSQL vs SQL trade-offs."

Expected Answer:  
"I’d start with EXPLAIN PLAN to identify bottlenecks. Add indexes on frequently filtered/joined columns (e.g., WHERE or JOIN keys). Partition the table by date or region to reduce scan size.  

If the data is unstructured or highly scalable (e.g., logs), I’d consider NoSQL (e.g., MongoDB) for flexibility, but SQL is better for complex joins and transactions.  

Optimization: Rewrite subqueries as JOINs, use covering indexes, and cache results with a tool like Redis if reads dominate."

Evaluation Guidelines
Scoring:  
Beginner: Understanding (0–3), Clarity (0–2)  

Intermediate: Application (0–4), Optimization (0–3)  

Challenging: Problem-Solving (0–5), Depth (0–5)

Weighting: Multiply scores by 20% per skill for the final average (e.g., 60% total for three skills).

Pass Threshold: 80% weighted average.

Coding Test Details
Tool: Use Replit or Google Colab for live coding.

Tasks:  
Python: Implement the Intermediate question (list merge).  

Machine Learning: Run the Challenging scenario code with a sample dataset.  

SQL: Execute the Intermediate query on a mock database (e.g., SQLite).

