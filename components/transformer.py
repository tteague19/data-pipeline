"""Module to transform the data."""
from abc import ABC, abstractmethod
from functools import reduce
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from components.extractor import LocalExtractor

class DataTransformer(ABC):
    """Abstract class to transform data."""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def transform_data(self) -> Dict[str, pd.DataFrame]:
        pass

class CSVDataTransformer(DataTransformer):

    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        transform_func_dict: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]]) \
            -> None:
        """Create an instance of the class."""
        self.data_dict = data_dict
        self.transform_func_dict = transform_func_dict

    def transform_data(self) -> Dict[str, pd.DataFrame]:
        """Transform the data according to the functions specified."""
        transformed_data_dict: Dict[str, pd.DataFrame] = {}
        for key, transform_func in self.transform_func_dict.items():
            transformed_data_dict[key] = rename_columns(
                self.data_dict[key])
            transformed_data_dict[key] = transform_func(transformed_data_dict[key])

        return transformed_data_dict


def create_overall_rating_mask(
    dataframe: pd.DataFrame,
    valid_ratings: Tuple[str, ...],
    rating_key: str) -> np.ndarray:
    mask = np.array(
        [
            dataframe[rating_key].values == value for value in valid_ratings
        ])
    mask = ~np.any(mask, axis=0)

    assert len(mask) == len(dataframe)
    return mask

def create_score_mask(
    dataframe: pd.DataFrame,
    measure_id: str,
    measure_key: str,
    score_key: str,
    invalid_values: Tuple[str, ...]) -> np.ndarray:

    invalid_entries_mask = np.any(
        np.array(
            [
            dataframe[score_key] == value for value in invalid_values
            ]),
        axis=0)

    mask = np.array(dataframe[measure_key] == measure_id) & invalid_entries_mask

    return mask


def rename_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe.columns = [
        "_".join(colname.lower().split(" ")) for colname in dataframe.columns
    ]

    return dataframe

def transform_care_dataframe(care_dataframe: pd.DataFrame) -> pd.DataFrame:

    # TODO: Create a class for this
    used_columns = [
        "facility_id",
        "measure_id",
        "measure_name",
        "score",
        "sample",
        "start_date",
        "end_date"
    ]
    measure_col = "measure_id"
    data_cols = ["score", "sample"]
    relevant_measure_ids = ["OP_31", "OP_22"]
    invalid_row_vals = ("Not Available",)
    dummy_val = -1

    # TODO: Extract into a function
    subset_dataframe = care_dataframe.copy()[used_columns]
    masks = [
            np.array(subset_dataframe[measure_col] == col)
            for col in relevant_measure_ids
    ]
    masks = reduce(lambda a, b: a | b, masks)
    subset_dataframe = subset_dataframe.copy().loc[masks, :]

    # TODO: Extract into a function
    for data_col in data_cols:
        for measure_id in relevant_measure_ids:
            mask = create_score_mask(
                dataframe=subset_dataframe,
                measure_id=measure_id,
                measure_key=measure_col,
                score_key=data_col,
                invalid_values=invalid_row_vals)
            subset_dataframe.loc[mask, data_col] = dummy_val
        subset_dataframe[data_col] = subset_dataframe.copy()[data_col].astype(int)

    return subset_dataframe

def transform_hospital_dataframe(
    hospital_dataframe: pd.DataFrame) -> pd.DataFrame:

    # TODO: Create a class for this.
    rating_key = "hospital_overall_rating"
    valid_ratings = ("1", "2", "3", "4", "5")
    dummy_val = 0
    used_columns = [
        "facility_id",
        "facility_name",
        "address",
        "city",
        "state",
        "zip_code",
        "county_name",
        "phone_number",
        "hospital_overall_rating"
    ]

    subset_dataframe = hospital_dataframe.copy()[used_columns]
    # We encode hospitals that do not contain an overall rating with
    # a rating of zero, which is outside the valid set of ratings of
    # {1, 2, 3, 4, 5}. We want to keep these hospitals in the
    # data, though, since we might be interested in creating a query
    # that involves information about the hospital that is present.
    ratings_mask = create_overall_rating_mask(
        dataframe=subset_dataframe,
        rating_key=rating_key,
        valid_ratings=valid_ratings)
    subset_dataframe.loc[ratings_mask, rating_key] = dummy_val

    return subset_dataframe

def main(file_dict: Dict[str, Path]) -> None:
    """
    Demonstrate a simple usage of the class.

    :param file_dict: A dictionary that maps a descriptive key to the
        full path of a file that contains data
    :type file_dict: Dict[str, Path]
    """
    extractor = LocalExtractor(file_dict=file_dict)
    data_dict = extractor.load_data()

    transform_func_dict = {
        "care": transform_care_dataframe,
        "hospital": transform_hospital_dataframe
    }

    data_transformer = CSVDataTransformer(
        data_dict=data_dict, transform_func_dict=transform_func_dict)

    transformed_data_dict = data_transformer.transform_data()


if __name__ == "__main__":
    file_dict = {
        "hospital": Path.cwd().joinpath(
            "data", "Hospital_General_Information.csv"),
        "care": Path.cwd().joinpath(
            "data", "Timely_and_Effective_Care-Hospital.csv")
    }

    main(file_dict=file_dict)