"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.19.12
"""
import pandas as pd

def clean_loan_data(data: pd.DataFrame) -> pd.DataFrame:
    # Drop unnecessary identifier column
    data = data.drop(columns=["LoanID"])

    # Convert categorical string columns to category dtype
    categorical_cols = [
        "Education", "EmploymentType", "MaritalStatus",
        "HasMortgage", "HasDependents", "LoanPurpose", "HasCoSigner"
    ]
    for col in categorical_cols:
        data[col] = data[col].astype("category")

    # Encode boolean-like columns ("Yes"/"No") as 1/0
    bool_map = {"Yes": 1, "No": 0}
    for col in ["HasMortgage", "HasDependents", "HasCoSigner"]:
        data[col] = data[col].map(bool_map)

    # Optional: Normalize column names (lowercase, underscores)
    data.columns = [col.lower().strip().replace(" ", "_") for col in data.columns]

    return data