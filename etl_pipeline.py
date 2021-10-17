"""Module to construct an ETL pipeline."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import pandas as pd

from components.extractor import Extractor, LocalExtractor
from components.transformer import DataTransformer, CSVDataTransformer
from components.transformer import transform_care_dataframe
from components.transformer import transform_hospital_dataframe
from components.loader import Loader, HospitalDataLoader


@dataclass
class ExtractorInput:
    """Class we pass to the methods of ETLPipeline instantiations"""
    file_dict: Dict[str, Path]


@dataclass
class TransformerInput:
    """Class we pass to the methods of ETLPipeline instantiations"""

    data_dict: Dict[str, pd.DataFrame]
    transform_func_dict: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]]


@dataclass
class LoaderInput:
    """Class we pass to the methods of ETLPipeline instantiations"""

    transformed_data_dict: Dict[str, pd.DataFrame]
    database_name_stem: str
    database_subdir: str


class ETLPipeline(ABC):
    """Abstract class that implements a factory for ETL pipeline components."""

    @abstractmethod
    def get_extractor(self, extractor_input: ExtractorInput) -> Extractor:
        """Retrieve a class that performs data extraction."""

    @abstractmethod
    def get_transformer(
        self, transformer_input: TransformerInput) -> DataTransformer:
        """Retrieve a class that performs data transformation."""

    @abstractmethod
    def get_loader(self, loader_input: LoaderInput) -> Loader:
        """Retrieve a class that creates a storage layer."""


class HospitalETLPipeline(ETLPipeline):
    """Class that creates an ETL pipeline specific to data about hospitals."""

    def get_extractor(
        self, extractor_input: ExtractorInput) -> Extractor:
        """
        Return an object to extract the hospital data.

        :param extractor_input: Input parameters necessary to
            construct an object that extracts the hospital data from
            a local directory
        :type extractor_input: ExtractorInput
        :return: An object to perform data extraction with the
            hospital data
        :rtype: Extractor
        """
        return LocalExtractor(file_dict=extractor_input.file_dict)

    def get_transformer(
        self, transformer_input: TransformerInput) -> DataTransformer:
        """
        Return an object to transform the hospital data.

        :param transformer_input: Input parameters necessary to
            construct an object that transforms the hospital data
        :type transformer_input: TransformerInput
        :return: An object to perform data transformation with the
            hospital data
        :rtype: DataTransformer
        """
        data_dict = transformer_input.data_dict
        transform_func_dict = transformer_input.transform_func_dict

        return CSVDataTransformer(
            data_dict=data_dict, transform_func_dict=transform_func_dict)

    def get_loader(self, loader_input: LoaderInput) -> Loader:
        """
        Return an object to place transformed hospital data in a storage layer.

        :param loader_input: Input parameters necessary to
            construct an object that loads the transformed hospital
            data into a storage layer
        :type loader_input: LoaderInput
        :return: An object to load the transformed hospital data into
            a storage layer
        :rtype: Loader
        """
        return HospitalDataLoader(
            transformed_data_dict=loader_input.transformed_data_dict,
            database_name_stem=loader_input.database_name_stem,
            database_subdir=loader_input.database_subdir)


def main(
        database_name_stem: str,
        database_subdir: str) -> None:
    """Build an ETL pipeline and create a database for hospital data."""
    pipeline = HospitalETLPipeline()
    file_dict  = {
        "hospital": Path.cwd().joinpath(
            "data", "Hospital_General_Information.csv"),
        "care": Path.cwd().joinpath(
            "data", "Timely_and_Effective_Care-Hospital.csv")
    }
    extractor = pipeline.get_extractor(ExtractorInput(file_dict=file_dict))
    data_dict = extractor.load_data()

    transform_func_dict = {
        "care": transform_care_dataframe,
        "hospital": transform_hospital_dataframe
    }
    transformer = pipeline.get_transformer(
        TransformerInput(
            data_dict=data_dict, transform_func_dict=transform_func_dict))
    transformed_data_dict = transformer.transform_data()

    loader = pipeline.get_loader(
        HospitalDataLoader(
            transformed_data_dict=transformed_data_dict,
            database_name_stem=database_name_stem,
            database_subdir=database_subdir))
    loader.create_storage_layer()


if __name__ == """__main__""":
    main(database_name_stem="hospital", database_subdir="databases")
