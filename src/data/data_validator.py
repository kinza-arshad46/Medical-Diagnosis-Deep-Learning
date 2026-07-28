"""
==========================================================
Medical Diagnosis AI
Dataset Validator
==========================================================

Author : Kinza Arshad

Description:
Validates Chest X-ray dataset before training.
==========================================================
"""

from pathlib import Path
import cv2

from src.utils.logger import logger
from src.utils.custom_exception import (
    MedicalDiagnosisException
)


class DataValidator:
    """
    Validate Chest X-ray dataset.
    """

    VALID_EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png"
    )

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.total_images = 0

        self.valid_images = 0

        self.corrupted_images = []

        self.class_distribution = {}

    def validate(self):

        try:

            if not self.dataset_path.exists():

                raise FileNotFoundError(
                    f"{self.dataset_path} not found."
                )

            class_folders = sorted(

                [

                    folder

                    for folder in self.dataset_path.iterdir()

                    if folder.is_dir()

                ]

            )

            if len(class_folders) == 0:

                raise ValueError(
                    "No class folders found."
                )

            logger.info(
                "Starting dataset validation..."
            )

            for class_folder in class_folders:

                image_count = 0

                for image_path in class_folder.iterdir():

                    if image_path.suffix.lower() not in self.VALID_EXTENSIONS:

                        continue

                    self.total_images += 1

                    image_count += 1

                    image = cv2.imread(
                        str(image_path)
                    )

                    if image is None:

                        self.corrupted_images.append(
                            str(image_path)
                        )

                    else:

                        self.valid_images += 1

                self.class_distribution[
                    class_folder.name
                ] = image_count

            logger.info(
                "Dataset validation completed."
            )

            return self.summary()

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def summary(self):

        return {

            "Total Images":
                self.total_images,

            "Valid Images":
                self.valid_images,

            "Corrupted Images":
                len(self.corrupted_images),

            "Classes":
                self.class_distribution,

            "Corrupted Files":
                self.corrupted_images

        }


if __name__ == "__main__":

    validator = DataValidator(
        "data/raw/train"
    )

    report = validator.validate()

    print("\n========== DATASET SUMMARY ==========\n")

    for key, value in report.items():

        print(f"{key} : {value}")