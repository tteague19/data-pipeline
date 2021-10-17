"""Module to construct an ETL pipeline."""
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from components.extractor import Extractor, LocalExtractor
from components.transformer import DataTransformer, CSVDataTransformer
from components.transformer import transform_care_dataframe
from components.transformer import transform_hospital_dataframe
from components.loader import Loader, HospitalDataLoader

class ETLPipeline(ABC):

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def get_extractor(self) -> Extractor:
        pass

    @abstractmethod
    def get_transformer(self) -> DataTransformer:
        pass

    @abstractmethod
    def get_loader(self) -> Loader:
        pass

class HospitalETLPipeline(ETLPipeline):

    def __init__(self, **kwargs) -> None:
        pass

    def get_extractor(self) -> Extractor:
        file_dict = {
            "hospital": Path.cwd().joinpath(
                "data", "Hospital_General_Information.csv"),
                "care": Path.cwd().joinpath(
                    "data", "Timely_and_Effective_Care-Hospital.csv")
        }

        return LocalExtractor(file_dict=file_dict)

    def get_transformer(self, **kwargs) -> DataTransformer:
        data_dict = kwargs["data_dict"]
        transform_func_dict = {
            "care": transform_care_dataframe,
            "hospital": transform_hospital_dataframe
        }

        return CSVDataTransformer(
            data_dict=data_dict, transform_func_dict=transform_func_dict)

    def get_loader(self, **kwargs) -> Loader:
        return HospitalDataLoader(
            transformed_data_dict=kwargs["transformed_data_dict"],
            database_name_stem=kwargs["database_name_stem"],
            database_subdir=kwargs["database_subdir"])


def main(
        database_name_stem: str,
        database_subdir: str) -> None:
    pipeline = HospitalETLPipeline()
    extractor = pipeline.get_extractor()
    data_dict = extractor.load_data()

    transformer = pipeline.get_transformer(data_dict=data_dict)
    transformed_data_dict = transformer.transform_data()

    loader = pipeline.get_loader(
        transformed_data_dict=transformed_data_dict,
        database_name_stem=database_name_stem,
        database_subdir=database_subdir)
    loader.create_storage_layer()


if __name__ == """__main__""":
    main(database_name_stem="hospital", database_subdir="databases")