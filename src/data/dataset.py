"""
==========================================================
Medical Diagnosis AI
Dataset Manager
==========================================================

Author : Kinza Arshad

Description:
Production-ready Dataset Manager with
Albumentations support.
==========================================================
"""

from pathlib import Path
import cv2

from torch.utils.data import Dataset, DataLoader

from src.utils.config_loader import ConfigLoader
from src.utils.logger import logger
from src.utils.custom_exception import MedicalDiagnosisException


class ChestXrayDataset(Dataset):

    """
    Custom Dataset for Chest X-ray Images.
    """

    VALID_EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png"
    )

    def __init__(self, root_dir, transform=None):

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

            class_name: idx

            for idx, class_name

            in enumerate(self.classes)

        }

        self.samples = []

        self._prepare_dataset()

    def _prepare_dataset(self):

        try:

            for class_name in self.classes:

                class_folder = self.root_dir / class_name

                if not class_folder.exists():

                    continue

                for image_path in class_folder.iterdir():

                    if image_path.suffix.lower() not in self.VALID_EXTENSIONS:

                        continue

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

                    f"Cannot read image : {image_path}"

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


class DatasetManager:

    """
    Creates datasets and dataloaders.
    """

    def __init__(self):

        self.config = ConfigLoader()

        self.batch_size = self.config.get(

            "training",

            "batch_size"

        )

        self.num_workers = self.config.get(

            "training",

            "num_workers"

        )

        self.shuffle = self.config.get(

            "training",

            "shuffle"

        )

    def create_dataset(

        self,

        root_dir,

        transform

    ):

        return ChestXrayDataset(

            root_dir=root_dir,

            transform=transform

        )

    def create_dataloader(

        self,

        dataset,

        shuffle=False

    ):

        return DataLoader(

            dataset,

            batch_size=self.batch_size,

            shuffle=shuffle,

            num_workers=self.num_workers,

            pin_memory=True

        )

    def get_dataloaders(

        self,

        train_transform,

        val_transform,

        test_transform

    ):

        try:

            train_dataset = self.create_dataset(

                self.config.get(

                    "paths",

                    "train_data"

                ),

                train_transform

            )

            val_dataset = self.create_dataset(

                self.config.get(

                    "paths",

                    "validation_data"

                ),

                val_transform

            )

            test_dataset = self.create_dataset(

                self.config.get(

                    "paths",

                    "test_data"

                ),

                test_transform

            )

            train_loader = self.create_dataloader(

                train_dataset,

                shuffle=True

            )

            val_loader = self.create_dataloader(

                val_dataset

            )

            test_loader = self.create_dataloader(

                test_dataset

            )

            logger.info(

                "Train, Validation and Test "

                "DataLoaders created successfully."

            )

            return (

                train_loader,

                val_loader,

                test_loader,

                train_dataset.classes

            )

        except Exception as e:

            raise MedicalDiagnosisException(e)


if __name__ == "__main__":

    from src.data.augmentation import ImageAugmentation

    augmenter = ImageAugmentation()

    dataset_manager = DatasetManager()

    train_loader, val_loader, test_loader, classes = (

        dataset_manager.get_dataloaders(

            augmenter.train_augmentation(),

            augmenter.validation_augmentation(),

            augmenter.test_augmentation()

        )

    )

    print("\nClasses :", classes)

    print("Train Images :", len(train_loader.dataset))

    print("Validation Images :", len(val_loader.dataset))

    print("Test Images :", len(test_loader.dataset))