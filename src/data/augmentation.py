"""
==========================================================
Medical Diagnosis AI
Advanced Image Augmentation
==========================================================

Author : Kinza Arshad

Description:
Creates advanced augmentation pipelines for
Chest X-ray images using Albumentations.
==========================================================
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.config_loader import ConfigLoader
from src.utils.logger import logger
from src.utils.custom_exception import MedicalDiagnosisException


class ImageAugmentation:
    """
    Creates augmentation pipelines for
    Train, Validation and Test datasets.
    """

    def __init__(self):

        self.config = ConfigLoader()

        self.image_size = self.config.get(
            "image",
            "image_size"
        )

        self.mean = self.config.get(
            "image",
            "normalize_mean"
        )

        self.std = self.config.get(
            "image",
            "normalize_std"
        )

    def train_augmentation(self):
        """
        Training augmentation pipeline.
        """

        try:

            transform = A.Compose([

                A.Resize(
                    self.image_size,
                    self.image_size
                ),

                A.HorizontalFlip(p=0.5),

                A.Rotate(
                    limit=15,
                    p=0.5
                ),

                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),

                A.GaussNoise(
                    p=0.2
                ),

                A.CLAHE(
                    clip_limit=2.0,
                    p=0.3
                ),

                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.05,
                    rotate_limit=10,
                    p=0.3
                ),

                A.Normalize(
                    mean=self.mean,
                    std=self.std
                ),

                ToTensorV2()

            ])

            logger.info(
                "Training augmentation created."
            )

            return transform

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def validation_augmentation(self):
        """
        Validation preprocessing.
        """

        try:

            transform = A.Compose([

                A.Resize(
                    self.image_size,
                    self.image_size
                ),

                A.Normalize(
                    mean=self.mean,
                    std=self.std
                ),

                ToTensorV2()

            ])

            logger.info(
                "Validation augmentation created."
            )

            return transform

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def test_augmentation(self):
        """
        Testing preprocessing.
        """

        try:

            transform = A.Compose([

                A.Resize(
                    self.image_size,
                    self.image_size
                ),

                A.Normalize(
                    mean=self.mean,
                    std=self.std
                ),

                ToTensorV2()

            ])

            logger.info(
                "Testing augmentation created."
            )

            return transform

        except Exception as e:

            raise MedicalDiagnosisException(e)


if __name__ == "__main__":

    augmenter = ImageAugmentation()

    train_transform = augmenter.train_augmentation()

    val_transform = augmenter.validation_augmentation()

    test_transform = augmenter.test_augmentation()

    print(train_transform)

    print(val_transform)

    print(test_transform)