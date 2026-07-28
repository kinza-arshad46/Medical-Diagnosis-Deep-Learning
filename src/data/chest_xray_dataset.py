"""
==========================================================
Medical Diagnosis AI
Custom Chest X-ray Dataset
==========================================================

Author : Kinza Arshad

Description:
Custom PyTorch Dataset with Albumentations support.
==========================================================
"""

from pathlib import Path
import cv2

from torch.utils.data import Dataset

from src.utils.logger import logger
from src.utils.custom_exception import (
    MedicalDiagnosisException
)


class ChestXrayDataset(Dataset):
    """
    Custom Dataset for Chest X-ray images.
    """

    def __init__(
        self,
        root_dir,
        transform=None
    ):

        self.root_dir = Path(root_dir)

        self.transform = transform

        self.classes = sorted(
            [
                folder.name
                for folder in self.root_dir.iterdir()
                if folder.is_dir()
            ]
        )

        self.class_to_idx = {
            class_name: index
            for index, class_name
            in enumerate(self.classes)
        }

        self.samples = []

        self._load_dataset()

    def _load_dataset(self):

        try:

            image_extensions = (
                ".jpg",
                ".jpeg",
                ".png"
            )

            for class_name in self.classes:

                class_folder = (
                    self.root_dir /
                    class_name
                )

                for image_path in class_folder.iterdir():

                    if image_path.suffix.lower() in image_extensions:

                        self.samples.append(

                            (
                                image_path,
                                self.class_to_idx[class_name]
                            )

                        )

            logger.info(
                f"{len(self.samples)} images loaded "
                f"from {self.root_dir}"
            )

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        try:

            image_path, label = self.samples[index]

            image = cv2.imread(str(image_path))

            if image is None:

                raise ValueError(
                    f"Cannot read image: {image_path}"
                )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            if self.transform:

                augmented = self.transform(
                    image=image
                )

                image = augmented["image"]

            return {

                "image": image,

                "label": label,

                "path": str(image_path)

            }

        except Exception as e:

            raise MedicalDiagnosisException(e)