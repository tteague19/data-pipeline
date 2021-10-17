"""Module to transform the data."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from components.extractor import LocalExtractor

@dataclass
class HospitalTransformParams:
    """Contain parameters relevant to transforming the hospital dataframe"""

    rating_key: str = "hospital_overall_rating"
    valid_ratings: Tuple[str, ...] = ("1", "2", "3", "4", "5")
    dummy_val: int = 0
    used_columns: Tuple[str, ...] = (
        "facility_id",
        "facility_name",
        "address",
        "city",
        "state",
        "zip_code",
        "county_name",
        "phone_number",
        "hospital_overall_rating"
    )

@dataclass
class CareTransformParams:
    """Contain parameters relevant to transforming the care dataframe"""

    used_columns: Tuple[str, ...] = (
        "facility_id",
        "measure_id",
        "measure_name",
        "score",
        "sample",
        "start_date",
        "end_date"
    )
    measure_col: str = "measure_id"
    data_cols: Tuple[str, ...] = ("score", "sample")
    relevant_measure_ids: Tuple[str, ...] = ("OP_31", "OP_22")
    invalid_row_vals: Tuple[str, ...] = ("Not Available",)
    dummy_val: int = -1

class DataTransformer(ABC):
    """Abstract class to transform data."""

    @abstractmethod
    def __init__(self) -> None:
        """Construct an instance of the class."""

    @abstractmethod
    def transform_data(self) -> Dict[str, pd.DataFrame]:
        """Transform the data to enable loading into a storage layer."""

class CSVDataTransformer(DataTransformer):
    """Class to transform data loaded from a CSV file."""

    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        transform_func_dict: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]]) \
            -> None:
        """Create an instance of the class."""
        super().__init__()
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
    """
    Create a mask that selects rows of the dataframe that lack valid data.

    :param dataframe: A dataframe that contains all rows from the
        original dataset
    :type dataframe: pd.DataFrame
    :param valid_ratings: A set of values that we consider valid. Rows
        that contain these values will be masked.
    :type valid_ratings: Tuple[str, ...]
    :param rating_key: The name of the column that contains the
        hospital ratings
    :type rating_key: str
    :return: A boolean array we can use to mask valid values
    :rtype: np.ndarray
    """
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
    """
    Create a mask that

    :param dataframe: A dataframe that contains all rows from the
        original dataset
    :type dataframe: pd.DataFrame
    :param measure_id: The string representation of a measure we wish
        to include in the mask
    :type measure_id: str
    :param measure_key: The name of the column that contains various
        measure IDs
    :type measure_key: str
    :param score_key: The name of the column that contains score
        values
    :type score_key: str
    :param invalid_values: Values in the score column we consider
        invalid
    :type invalid_values: Tuple[str, ...]
    :return: A boolean array we can use to mask invalid entries in a
        dataframe
    :rtype: np.ndarray
    """
    invalid_entries_mask = np.any(
        np.array(
            [
            dataframe[score_key] == value for value in invalid_values
            ]),
        axis=0)

    mask = np.array(dataframe[measure_key] == measure_id)  # nd.array
    mask = mask & invalid_entries_mask

    return mask


def rename_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to be lowercase and separated by an underscore.

    :param dataframe: A dataframe whose columns we wish to rename
    :type dataframe: pd.DataFrame
    :return: The input dataframe with renamed columns
    :rtype: pd.DataFrame
    """
    dataframe.columns = [
        "_".join(colname.lower().split(" ")) for colname in dataframe.columns
    ]

    return dataframe

def transform_care_dataframe(care_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a dataframe that describes care information for various hospitals.

    :param care_dataframe: A dataframe that describes care information for
        various hospitals
    :type care_dataframe: pd.DataFrame
    :return: The input dataframe transformed to enable further processing
    :rtype: pd.DataFrame
    """
    transform_input = CareTransformParams()

    # TODO: Extract into a function
    subset_dataframe = care_dataframe.copy()[list(transform_input.used_columns)]
    masks = [
            np.array(subset_dataframe[transform_input.measure_col] == col)
            for col in transform_input.relevant_measure_ids
    ]
    reduced_masks = reduce(lambda a, b: a | b, masks)  # type: np.ndarray
    subset_dataframe = subset_dataframe.copy().loc[reduced_masks, :]

    # TODO: Extract into a function
    for data_col in transform_input.data_cols:
        for measure_id in transform_input.relevant_measure_ids:
            mask = create_score_mask(
                dataframe=subset_dataframe,
                measure_id=measure_id,
                measure_key=transform_input.measure_col,
                score_key=data_col,
                invalid_values=transform_input.invalid_row_vals)
            subset_dataframe.loc[mask, data_col] = transform_input.dummy_val
        subset_dataframe[data_col] = subset_dataframe.copy()[data_col].astype(int)

    return subset_dataframe

def transform_hospital_dataframe(
    hospital_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a dataframe of information about various hospitals.

    :param hospital_dataframe: A dataframe that contains information
        about various hospitals
    :type hospital_dataframe: pd.DataFrame
    :return: A processed version of the input dataframe that enables
        further processing
    :rtype: pd.DataFrame
    """
    transform_input = HospitalTransformParams()

    subset_dataframe = hospital_dataframe.copy()[list(transform_input.used_columns)]

    # We encode hospitals that do not contain an overall rating with
    # a rating of zero, which is outside the valid set of ratings of
    # {1, 2, 3, 4, 5}. We want to keep these hospitals in the
    # data, though, since we might be interested in creating a query
    # that involves information about the hospital that is present.
    ratings_mask = create_overall_rating_mask(
        dataframe=subset_dataframe,
        rating_key=transform_input.rating_key,
        valid_ratings=transform_input.valid_ratings)
    subset_dataframe.loc[ratings_mask, transform_input.rating_key] = transform_input.dummy_val

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
    data_desc_to_path_dict = {
        "hospital": Path.cwd().joinpath(
            "data", "Hospital_General_Information.csv"),
        "care": Path.cwd().joinpath(
            "data", "Timely_and_Effective_Care-Hospital.csv")
    }

    main(file_dict=data_desc_to_path_dict)
