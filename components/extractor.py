"""Module to read data."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import pandas as pd


class Extractor(ABC):
    """Abstract class to load data from a source location."""

    @abstractmethod
    def __init__(self, **kwargs: Dict[str, Path]) -> None:
        """Create an instance of the class."""
        pass

    @abstractmethod
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load the data."""
        pass


class LocalExtractor(Extractor):
    """
    A class to load data files from a local directory,

    :param file_dict: A dictionary that maps a descriptive key to the
        full path of a data file
    :type file_dict: Dict[str, Path]
    """

    def __init__(self, **kwargs: Dict[str, Path]) -> None:
        """Create an instance of the class."""
        self.file_dict = kwargs["file_dict"]  # type: Dict[str, Path]

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load the data from its specified locations.

        :return: A dictionary that maps a descriptive key to a pandas
            DataFrame
        :rtype: Dict[str, pd.DataFrame]
        """
        data_dict: Dict[str, pd.DataFrame] = {}
        for file_key, file_path in self.file_dict.items():
            data_dict[file_key] = pd.read_csv(file_path)

        return data_dict


def main(file_dict: Dict[str, Path]) -> None:
    """
    Demonstrate a simple usage of the class.

    :param file_dict: A dictionary that maps a descriptive key to the
        full path of a file that contains data
    :type file_dict: Dict[str, Path]
    """
    extractor = LocalExtractor(file_dict=file_dict)
    data_dict = extractor.load_data()

    print(data_dict["care"].head())


if __name__ == "__main__":
    file_dict = {
        "hospital": Path.cwd().joinpath(
            "data", "Hospital_General_Information.csv"),
        "care": Path.cwd().joinpath(
            "data", "Timely_and_Effective_Care-Hospital.csv")
    }

    main(file_dict=file_dict)
