# Customer Segmentation Using K-Means Clustering: Project Report

## 1. Introduction

Customer segmentation is the process of dividing a company's customers into groups that share similar characteristics. It helps businesses understand who their customers are, target marketing campaigns more effectively, and allocate resources where they will have the greatest impact.

This project segments mall customers using **K-Means clustering**, an unsupervised machine learning algorithm. The analysis is based on two features: **Annual Income** and **Spending Score**. Once the clusters are formed, they are further analyzed by age and gender.

---

## 2. Why K-Means Clustering?

K-Means was chosen for the following reasons:

| Reason | Explanation |
|---|---|
| **Unsupervised learning** | The dataset has no predefined customer labels, so the algorithm must discover natural groupings on its own. K-Means is designed for exactly this situation. |
| **Simplicity and interpretability** | Each cluster is represented by a centroid (an average point), which makes results easy to explain to non-technical stakeholders such as marketing teams. |
| **Efficiency** | K-Means scales well to larger datasets, and its computational cost grows linearly with the number of samples. |
| **Suitability for numeric features** | Income and spending score are continuous numeric variables, which K-Means handles naturally using Euclidean distance. |
| **Well-established evaluation tools** | The Elbow Method and Silhouette Score provide clear, objective ways to choose the number of clusters. |
| **Proven business use** | K-Means is a standard, widely used technique in customer segmentation, making the results comparable to industry practice. |

**Limitations to keep in mind:** K-Means assumes roughly spherical clusters of similar size, requires the number of clusters (K) to be chosen in advance, and is sensitive to outliers and to feature scaling. These issues were addressed in the project through standardization, the Elbow and Silhouette methods, and the use of `random_state` and `n_init` for reproducibility.

---

## 3. Dataset

The dataset is `Mall_Customers.csv`, which typically contains the following columns:

| Column | Description |
|---|---|
| `CustomerID` | Unique identifier for each customer |
| `Gender` | Male or Female |
| `Age` | Customer's age in years |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Score assigned by the mall based on customer behavior and spending |

---

## 4. Line-by-Line Code Explanation

### 4.1 Imports

```python
import pandas as pd
```
Imports **pandas**, the library used to load, store, and manipulate tabular data in DataFrames.

```python
import matplotlib.pyplot as plt
```
Imports **matplotlib's pyplot** module, used to create charts and graphs.

```python
from sklearn.preprocessing import StandardScaler
```
Imports the **StandardScaler**, which standardizes features to have mean 0 and standard deviation 1.

```python
from sklearn.cluster import KMeans
```
Imports the **KMeans** class from scikit-learn, the core clustering algorithm.

```python
from sklearn.metrics import silhouette_score
```
Imports the **silhouette_score** function, which measures how well-separated the clusters are.

---

### 4.2 Loading the Data (Section 1)

```python
df = pd.read_csv(r'D:\IMPORTANT FILES\vs code programs\Mall-Customer-Analysis\data\Mall_Customers.csv')
```
Reads the CSV file into a pandas DataFrame named `df`. The `r` prefix creates a *raw string*, so Windows backslashes are not treated as escape characters.

```python
print("First 5 Rows:")
print(df.head())
```
Prints the label and then the first five rows, giving a quick preview of the data.

```python
print("\nDataset Information:")
print(df.info())
```
`df.info()` shows the number of rows, column names, data types, and non-null counts. It is used to confirm the data loaded correctly.

```python
print("\nMissing Values:")
print(df.isnull().sum())
```
`df.isnull()` marks every missing value as `True`, and `.sum()` counts them per column. This step checks data quality; K-Means cannot work with missing values, so this check is essential.

---

### 4.3 Feature Selection (Section 3)

```python
X = df[
    [
        'Annual Income (k$)',
        'Spending Score (1-100)'
    ]
]
```
Selects only the two features used for clustering and stores them in `X`. Using two features keeps the model simple and makes the clusters easy to visualize on a 2D scatter plot. Gender and age are deliberately excluded from the clustering itself and are used later for profiling.

---

### 4.4 Feature Scaling (Section 4)

```python
scaler = StandardScaler()
```
Creates a `StandardScaler` object.

```python
X_scaled = scaler.fit_transform(X)
```
Computes the mean and standard deviation of each feature and transforms the values using:

$$z = \frac{x - \mu}{\sigma}$$

This is important because income (ranging roughly 15–137) and spending score (1–100) are on different scales. Without scaling, the feature with the larger range would dominate the distance calculations that K-Means relies on.

---

### 4.5 Finding the Optimal K (Section 5)

```python
inertia_values = []
silhouette_values = []
```
Creates two empty lists to store the inertia (Elbow Method) and silhouette score for each value of K.

```python
k_values = range(2, 11)
```
Defines the range of cluster counts to test: K = 2 through 10.

```python
for k in k_values:
```
Loops over each candidate value of K.

```python
model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)
```
Creates a KMeans model with `k` clusters. `random_state=42` makes results reproducible, and `n_init=10` runs the algorithm 10 times with different random centroid starts, keeping the best result to avoid poor local optima.

```python
labels = model.fit_predict(X_scaled)
```
Fits the model to the scaled data and returns the cluster label assigned to each customer.

```python
inertia_values.append(model.inertia_)
```
Stores the **inertia**, the sum of squared distances between each point and its assigned centroid. Lower inertia means tighter clusters.

```python
silhouette_values.append(
    silhouette_score(X_scaled, labels)
)
```
Computes the **silhouette score**, which ranges from -1 to 1, for this value of K. Higher values indicate better-separated clusters.

#### Elbow Method Plot

```python
plt.figure(figsize=(8, 5))
```
Creates a new figure 8 inches wide and 5 inches tall.

```python
plt.plot(list(k_values), inertia_values, marker='o')
```
Plots inertia against K, with a circle marker at each point. The "elbow," where the curve's rate of decrease sharply slows, suggests a good K.

```python
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid()
plt.show()
```
Labels the axes, adds a title, adds a grid for readability, and displays the plot.

#### Silhouette Plot

```python
plt.plot(list(k_values), silhouette_values, marker='o')
```
Plots silhouette score against K. The K with the highest score is generally the best choice.

The remaining lines in this block perform the same labeling and display steps as the Elbow plot.

---

### 4.6 Applying K-Means (Section 6)

```python
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
```
Creates the final model with **K = 5**, the value selected from the evaluation plots.

```python
df['Cluster'] = kmeans.fit_predict(X_scaled)
```
Fits the model and adds a new `Cluster` column to the DataFrame, assigning each customer to one of five groups (0–4).

```python
print("\nClustering Results:")
print(df.head())
```
Displays the first five rows with their cluster assignments.

---

### 4.7 Cluster Analysis (Section 7)

```python
cluster_analysis = df.groupby('Cluster').agg(
    Customers=('CustomerID', 'count'),
    Average_Age=('Age', 'mean'),
    Average_Income=('Annual Income (k$)', 'mean'),
    Average_Spending=('Spending Score (1-100)', 'mean')
).round(2)
```
Groups customers by cluster and calculates:
- `Customers`: number of customers in the cluster
- `Average_Age`, `Average_Income`, `Average_Spending`: the mean value of each attribute
- `.round(2)`: rounds results to two decimal places

```python
print(cluster_analysis)
```
Displays the summary table.

---

### 4.8 Age Group Analysis (Section 8)

```python
df['Age_Group'] = pd.cut(
    df['Age'],
    bins=[17, 25, 35, 45, 55, 70],
    labels=['18-25', '26-35', '36-45', '46-55', '56-70']
)
```
Converts continuous ages into categorical bins. `pd.cut` assigns each age to the interval it falls into. The bin edges are left-open and right-closed, so the first bin covers ages 18 through 25.

```python
age_analysis = pd.crosstab(df['Cluster'], df['Age_Group'])
```
Creates a contingency table counting how many customers from each age group appear in each cluster.

```python
print(age_analysis)
```
Displays the table.

---

### 4.9 Gender Analysis (Section 9)

```python
gender_analysis = pd.crosstab(df['Cluster'], df['Gender'])
```
Counts male and female customers in each cluster.

```python
print(gender_analysis)
```
Displays the gender distribution.

---

### 4.10 Income and Spending by Age (Section 10)

```python
age_income_spending = df.groupby('Age_Group', observed=False).agg(
    Customers=('CustomerID', 'count'),
    Average_Income=('Annual Income (k$)', 'mean'),
    Average_Spending=('Spending Score (1-100)', 'mean')
).round(2)
```
Groups customers by age group and computes the count and average income and spending for each. `observed=False` keeps age groups that have no customers in the output, which is the default behavior in future pandas versions and avoids warnings.

```python
print(age_income_spending)
```
Displays the results.

---

### 4.11 Income and Spending by Gender (Section 11)

```python
gender_income_spending = df.groupby('Gender').agg(
    Customers=('CustomerID', 'count'),
    Average_Income=('Annual Income (k$)', 'mean'),
    Average_Spending=('Spending Score (1-100)', 'mean')
).round(2)
```
Performs the same aggregation as above but grouped by gender.

```python
print(gender_income_spending)
```
Displays the results.

---

### 4.12 Cluster Visualization (Section 12)

```python
plt.figure(figsize=(10, 6))
```
Creates a larger figure for the scatter plot.

```python
plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],
    cmap='viridis',
    s=60
)
```
Draws a scatter plot with income on the x-axis and spending score on the y-axis. Each point is colored by its cluster label using the `viridis` colormap, and `s=60` sets the marker size.

```python
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.colorbar(label="Cluster")
plt.show()
```
Adds axis labels, a title, a colorbar showing the cluster scale, and displays the chart.

---

### 4.13 Age Group Bar Chart (Section 13)

```python
age_analysis.plot(kind='bar', figsize=(10, 6))
```
Creates a grouped bar chart from the age crosstab, with one group of bars per cluster.

```python
plt.title("Customer Age Groups by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```
Adds labels, keeps the x-axis tick labels horizontal, adjusts spacing so nothing is clipped, and displays the chart.

---

### 4.14 Gender Bar Chart (Section 14)

```python
gender_analysis.plot(kind='bar', figsize=(8, 5))
```
Creates a grouped bar chart showing male and female counts per cluster.

The following lines add labels and display the chart in the same way as Section 13.

---

### 4.15 Age and Spending Bar Chart (Section 15)

```python
age_income_spending[['Average_Income', 'Average_Spending']].plot(kind='bar', figsize=(10, 5))
```
Selects the two average columns from the age table and plots them as grouped bars for each age group.

```python
plt.title("Average Income and Spending by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```
Adds labels and displays the chart.

---

### 4.16 Gender and Spending Bar Chart (Section 16)

```python
gender_income_spending[['Average_Income', 'Average_Spending']].plot(kind='bar', figsize=(8, 5))
```
Plots average income and spending for male and female customers.

```python
plt.title("Average Income and Spending by Gender")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```
Adds labels and displays the chart.

---

### 4.17 Final Results (Section 17)

```python
print("\nFinal Silhouette Score:")
print(silhouette_score(X_scaled, df['Cluster']))
```
Calculates and prints the silhouette score for the final five-cluster model. This is the overall measure of clustering quality.

```python
print("\nProject Completed Successfully!")
```
Prints a confirmation message indicating the script finished.

---

## 5. Results

> **Note:** The exact numbers below depend on the dataset version and on the output of your run. The values shown are the typical results for the standard Mall Customers dataset (200 customers). Please replace them with the values printed by your script if they differ.

### 5.1 Optimal K Selection

- **Elbow Method:** Inertia decreases rapidly from K = 2 to about K = 5, then flattens. This bend indicates that five clusters capture most of the structure without unnecessary complexity.
- **Silhouette Method:** The silhouette score peaks at or near K = 5 (approximately 0.55), indicating well-separated and cohesive clusters.

Together, both methods support the choice of **K = 5**.

### 5.2 Cluster Summary

| Cluster | Customers | Avg. Age | Avg. Income (k$) | Avg. Spending | Typical Profile |
|---|---|---|---|---|---|
| 0 | ~35 | ~40 | ~88 | ~17 | High income, low spending (cautious spenders) |
| 1 | ~40 | ~33 | ~86 | ~82 | High income, high spending (ideal customers) |
| 2 | ~23 | ~25 | ~25 | ~79 | Low income, high spending (young enthusiastic buyers) |
| 3 | ~22 | ~45 | ~26 | ~20 | Low income, low spending (budget-conscious) |
| 4 | ~80 | ~43 | ~55 | ~50 | Average income, average spending (moderate customers) |

*(Values are approximate and should be verified against your script output.)*

### 5.3 Interpretation of Segments

1. **Premium Customers (High Income, High Spending):** Most valuable. Target with loyalty programs, premium products, and personalized offers.
2. **Careful Customers (High Income, Low Spending):** Have the capacity to spend but do not. Target with promotions, value messaging, or better product discovery.
3. **Young Enthusiasts (Low Income, High Spending):** Engaged and spend freely despite limited income. Suitable for trend-driven products and social media campaigns.
4. **Budget Customers (Low Income, Low Spending):** Price-sensitive. Target with discounts and essential products.
5. **Average Customers (Middle Income and Spending):** The largest group. Target with general campaigns and upselling opportunities.

### 5.4 Demographic Findings

- **Age:** Younger customers tend to have higher spending scores, while older customers are more represented in the low-spending clusters.
- **Gender:** Female customers make up a larger share of the customer base, and their distribution across clusters is fairly even.
- **Income vs. Spending:** Income and spending score do not have a strong direct relationship. Some high-income customers spend little, and some low-income customers spend heavily. This is the main reason clustering on both features reveals more than either feature alone.

### 5.5 Model Quality

- **Final Silhouette Score:** approximately **0.55**, which is considered good for real-world customer data.
- Clusters are well separated on the income-spending plot, and the five groups are distinct and interpretable.

---

## 6. Conclusion

This project successfully segmented mall customers into five distinct groups using K-Means clustering. The Elbow and Silhouette methods both supported K = 5 as the optimal number of clusters. The resulting segments reveal clear differences in income, spending behavior, age, and gender, providing actionable insights for targeted marketing strategies.

K-Means was an appropriate choice because the problem is unsupervised, the features are numeric, the method is efficient and interpretable, and it is a widely accepted standard for customer segmentation.

### Possible Future Improvements
- Include age and gender directly in the clustering model (with appropriate encoding and scaling).
- Compare K-Means with other algorithms such as Hierarchical Clustering or DBSCAN.
- Use Principal Component Analysis (PCA) if more features are added.
- Track segment behavior over time to measure the impact of marketing campaigns.

