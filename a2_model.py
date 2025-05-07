import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Load the dataset
file_path = 'D:\Project expo\Insurance sample data\A2 Data.xlsx'  
data = pd.read_excel(file_path)

# Select specific attributes
selected_data = data[['Job Family', 'Union']]

# Convert nominal data to binomial
binomial_data = pd.get_dummies(selected_data, columns=['Job Family', 'Union'])

# Apply FP-Growth algorithm
frequent_itemsets = fpgrowth(
    binomial_data,
    min_support=0.05,
    use_colnames=True,
    max_len=None
)

# Create association rules
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.7,
    support_only=False
)

# Display the association rules
print(rules)
