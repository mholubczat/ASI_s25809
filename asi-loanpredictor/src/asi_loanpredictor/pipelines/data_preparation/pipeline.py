"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.19.12
"""

from kedro.pipeline import Pipeline, node
from .nodes import clean_loan_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=clean_loan_data,
                inputs="loans",
                outputs="cleaned_loan_data",
                name="clean_loan_data_node"
            ),
        ]
    )