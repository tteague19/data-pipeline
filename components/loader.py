"Module to create a storage layer for our data."
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd
import sqlite3

from components.extractor import LocalExtractor
from components.transformer import CSVDataTransformer
from components.transformer import transform_care_dataframe, transform_hospital_dataframe

class Loader(ABC):

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        """Construct an instance of the class."""
        pass

class HospitalDataLoader(Loader):

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        self.transformed_data_dict = kwargs["transformed_data_dict"]  # type: Dict[str, pd.DataFrame]
        self.database_name_stem = kwargs["database_name_stem"]  # type: str
        self.database_subdir = kwargs["database_subdir"]  # type: str

    def _join_dataframes(self, index_name: str) -> pd.DataFrame:
        for dataframe in self.transformed_data_dict.values():
            dataframe.set_index(index_name, inplace=True)

        joined_dataframe = pd.merge(
            self.transformed_data_dict["care"],
            self.transformed_data_dict["hospital"],
            left_index=True,
            right_index=True,
            how="left",
            validate="many_to_one")

        return joined_dataframe

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
        joined_dataframe = self._join_dataframes(index_name="facility_id")
        self._write_to_database(joined_dataframe=joined_dataframe)


def main(
    file_dict: Dict[str, Path],
    transform_func_dict: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]],
    database_name_stem: str,
    database_subdir: str) -> None:
    extractor = LocalExtractor(file_dict=file_dict)
    data_dict = extractor.load_data()

    data_transformer = CSVDataTransformer(
        data_dict=data_dict, transform_func_dict=transform_func_dict)
    transformed_data_dict = data_transformer.transform_data()

    loader = HospitalDataLoader(transformed_data_dict=transformed_data_dict)
    loader.create_storage_layer()

    conn = sqlite3.connect(Path.cwd().joinpath(
        database_subdir, f"{database_name_stem}.db"))
    cur = conn.cursor()
    cur.execute(f"SELECT * from {database_name_stem}")
    rows = cur.fetchall()
    for row in rows:
        print(row)


if __name__ == "__main__":
    file_dict = {
        "hospital": Path.cwd().joinpath(
            "data", "Hospital_General_Information.csv"),
        "care": Path.cwd().joinpath(
            "data", "Timely_and_Effective_Care-Hospital.csv")
    }

    transform_func_dict = {
        "care": transform_care_dataframe,
        "hospital": transform_hospital_dataframe
    }

    main(
        file_dict=file_dict,
        transform_func_dict=transform_func_dict,
        database_name_stem="hospitals",
        database_subdir="databases")


# lower_bounds = [0, 0]
# upper_bounds = [100, np.inf]

# # TODO: Move validation to the loader.
# for measure_id, min_val, max_val in zip(relevant_measure_ids, lower_bounds, upper_bounds):
#     sub_cond1 = subset_dataframe.loc[subset_dataframe[measure_col] == measure_id, score_col] > max_val
#     sub_cond2 = subset_dataframe.loc[subset_dataframe[measure_col] == measure_id, score_col] < min_val
#     cond1 = all(subset_dataframe.loc[sub_cond1 & sub_cond2, score_col] == -1)
#     cond2 = all(subset_dataframe.loc[subset_dataframe[measure_col] == measure_id, score_col]) <= max_val
#     cond3 = all(subset_dataframe.loc[subset_dataframe[measure_col] == measure_id, score_col]) >= min_val
#     assert cond1 or cond2 or cond3