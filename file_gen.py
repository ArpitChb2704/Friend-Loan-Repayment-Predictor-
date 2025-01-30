import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Define the number of entries
num_entries = 1000

# Generate synthetic data
data = {
    "Amount_Lent_INR": np.random.randint(10, 50000, num_entries),  # Random amounts between ₹1000 and ₹50000
    "Relationship_with_Borrower": np.random.choice(["Close Friend", "Family", "Acquaintance"], num_entries),
    "Repayment_History": np.random.choice(["Yes", "No"], num_entries, p=[0.7, 0.3]),  # 70% Yes, 30% No
    "Borrower_Financial_Situation": np.random.choice(["Good", "Average", "Poor"], num_entries, p=[0.5, 0.3, 0.2]),
    "Social_Trust_Factors": np.random.randint(1, 11, num_entries),  # Random trust score between 1 and 10
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define repayment outcome based on rules
def determine_repayment_outcome(row):
    if row["Relationship_with_Borrower"] == "Close Friend" and row["Repayment_History"] == "Yes" and row["Borrower_Financial_Situation"] == "Good":
        return "Yes"
    elif row["Relationship_with_Borrower"] == "Family" and row["Social_Trust_Factors"] >= 7:
        return "Yes"
    elif row["Borrower_Financial_Situation"] == "Poor" and row["Social_Trust_Factors"] <= 4:
        return "No"
    else:
        return np.random.choice(["Yes", "No"], p=[0.6, 0.4])  # Default probability

df["Repayment_Outcome"] = df.apply(determine_repayment_outcome, axis=1)

# Save the dataset to a CSV file
df.to_csv("lending_prediction_dataset.csv", index=False)

print("Dataset generated and saved as 'lending_prediction_dataset.csv'")