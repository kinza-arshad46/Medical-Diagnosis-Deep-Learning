"""
==========================================================
Medical Diagnosis AI
Data Pipeline
==========================================================

Author : Kinza Arshad

Description:
Centralized Data Pipeline for
Training, Validation and Testing.
==========================================================
"""

from src.data.dataset import DatasetManager
from src.data.augmentation import ImageAugmentation

from src.utils.logger import logger
from src.utils.custom_exception import (
    MedicalDiagnosisException
)


class DataPipeline:
    """
    Centralized Data Pipeline.
    """

    def __init__(self):

        self.dataset_manager = DatasetManager()

        self.augmentation = ImageAugmentation()

    def get_dataloaders(self):

        """
        Returns train, validation and
        test dataloaders.
        """

        try:

            train_transform = (

                self.augmentation.train_augmentation()

            )

            validation_transform = (

                self.augmentation.validation_augmentation()

            )

            test_transform = (

                self.augmentation.test_augmentation()

            )

            (

                train_loader,

                validation_loader,

                test_loader,

                class_names

            ) = self.dataset_manager.get_dataloaders(

                train_transform,

                validation_transform,

                test_transform

            )

            logger.info(

                "Data Pipeline initialized successfully."

            )

            return (

                train_loader,

                validation_loader,

                test_loader,

                class_names

            )

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def dataset_summary(self):

        """
        Prints dataset statistics.
        """

        try:

            (

                train_loader,

                validation_loader,

                test_loader,

                class_names

            ) = self.get_dataloaders()

            summary = {

                "Classes": class_names,

                "Number of Classes": len(class_names),

                "Training Images":
                    len(train_loader.dataset),

                "Validation Images":
                    len(validation_loader.dataset),

                "Testing Images":
                    len(test_loader.dataset)

            }

            logger.info(summary)

            return summary

        except Exception as e:

            raise MedicalDiagnosisException(e)


if __name__ == "__main__":

    pipeline = DataPipeline()

    summary = pipeline.dataset_summary()

    print("\n========== DATASET SUMMARY ==========\n")

    for key, value in summary.items():

        print(f"{key}: {value}")