"Module to create a storage layer for our data."
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Dict
import sqlite3

import numpy as np
import pandas as pd

from components.extractor import LocalExtractor
from components.transformer import CSVDataTransformer
from components.transformer import transform_care_dataframe
from components.transformer import transform_hospital_dataframe

@dataclass
class DataValidator:
    """A class to hold parameters relevant for validating data."""

    measure_ids: List[str]
    lower_bounds: List[float]
    upper_bounds: List[float]
    measure_column: str
    score_column: str
    dummy_vals: List[int]

class Loader(ABC):
    """An abstract class to load data into a storage layer."""

    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Construct an instance of the class."""

class HospitalDataLoader(Loader):
    """
    A class to load transformed hospital data into a storage layer

    :param transformed_data_dict: A dictionary that maps a descriptive
        string to dataframes that have been transformed according to
        a process we specified
    :type transformed_data_dict: Dict[str, pd.DataFrame]
    :param database_name_stem: The name of the database we will create
    :type database_name_stem: str
    :param database_subdir: The name of the sub-directory into which
        we will save the database we create
    :type database_subdir: str
    """

    def __init__(self, **kwargs: Any) -> None:
        """Create an instance of the class."""
        super().__init__()
        self.transformed_data_dict = kwargs["transformed_data_dict"]
        self.database_name_stem = kwargs["database_name_stem"]
        self.database_subdir = kwargs["database_subdir"]

    def _join_dataframes(self, index_name: str) -> pd.DataFrame:
        for dataframe in self.transformed_data_dict.values():
            validator = DataValidator(
                measure_ids=[],
                lower_bounds=[0, 0],
                upper_bounds=[100, np.inf],
                measure_column="measure_ids",
                score_column="score",
                dummy_vals=[0, -1])
            dataframe.set_index(index_name, inplace=True)

            self._validate_data(df=dataframe, data_validator=validator)

        joined_dataframe = pd.merge(
            self.transformed_data_dict["care"],
            self.transformed_data_dict["hospital"],
            left_index=True,
            right_index=True,
            how="left",
            validate="many_to_one")

        return joined_dataframe

    @staticmethod
    def _validate_data(
        df: pd.DataFrame, data_validator: DataValidator) -> None:
        meas_ids = data_validator.measure_ids
        lower_bds = data_validator.lower_bounds
        upper_bds = data_validator.upper_bounds
        meas_col = data_validator.measure_column
        score_col = data_validator.score_column
        vals = data_validator.dummy_vals

        gen = zip(meas_ids, lower_bds, upper_bds, vals)
        for meas_id, min_val, max_val, val in gen:
            sub_cond1 = df.loc[df[meas_col] == meas_id, score_col] > max_val
            sub_cond2 = df.loc[df[meas_col] == meas_id, score_col] < min_val
            cond1 = all(df.loc[sub_cond1 & sub_cond2, score_col] == val)
            cond2 = all(df.loc[df[meas_col] == meas_id, score_col]) <= max_val
            cond3 = all(df.loc[df[meas_col] == meas_id, score_col]) >= min_val
            assert cond1 or cond2 or cond3

    def _write_to_database(self, joined_dataframe: pd.DataFrame) -> None:
        database_path = Path.cwd().joinpath(
            self.database_subdir, f"{self.database_name_stem}.db")
        database_path.parents[0].mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(rf"{database_path.as_posix()}")
        joined_dataframe.to_sql(
            self.database_name_stem,
            con=conn,
            index=False,
            if_exists="replace")


    def create_storage_layer(self) -> None:
        """Create a storage layer specific to hospital data."""
        joined_dataframe = self._join_dataframes(index_name="facility_id")
        self._write_to_database(joined_dataframe=joined_dataframe)


def main(
    file_dict: Dict[str, Path],
    transform_func_dict: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]],
    database_name_stem: str,
    database_subdir: str) -> None:
    """Demonstrate the use of the HospitalDataLoader class."""
    extractor = LocalExtractor(file_dict=file_dict)
    data_dict = extractor.load_data()

    data_transformer = CSVDataTransformer(
        data_dict=data_dict, transform_func_dict=transform_func_dict)
    transformed_data_dict = data_transformer.transform_data()

    loader = HospitalDataLoader(
        transformed_data_dict=transformed_data_dict,
        database_name_stem=database_name_stem,
        database_subdir=database_subdir)
    loader.create_storage_layer()

    conn = sqlite3.connect(Path.cwd().joinpath(
        database_subdir, f"{database_name_stem}.db"))
    cur = conn.cursor()
    cur.execute(f"SELECT * from {database_name_stem}")
    rows = cur.fetchall()
    for row in rows:
        print(row)


if __name__ == "__main__":
    data_desc_to_path_dict = {
        "hospital": Path.cwd().joinpath(
            "data", "Hospital_General_Information.csv"),
        "care": Path.cwd().joinpath(
            "data", "Timely_and_Effective_Care-Hospital.csv")
    }

    data_desc_to_transform_func_dict = {
        "care": transform_care_dataframe,
        "hospital": transform_hospital_dataframe
    }

    main(
        file_dict=data_desc_to_path_dict,
        transform_func_dict=data_desc_to_transform_func_dict,
        database_name_stem="hospital",
        database_subdir="databases")
