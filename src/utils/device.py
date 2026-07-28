"""
==========================================================
Medical Diagnosis AI
Device Configuration
==========================================================

Author : Kinza Arshad

Description:
Automatically detects the best available device
(CPU or GPU) for training and inference.
==========================================================
"""

import torch

from src.utils.logger import logger
from src.utils.custom_exception import MedicalDiagnosisException


class DeviceManager:
    """
    Handles CPU / GPU selection.
    """

    @staticmethod
    def get_device():

        try:

            if torch.cuda.is_available():

                device = torch.device("cuda")

                logger.info(
                    f"GPU Detected: {torch.cuda.get_device_name(0)}"
                )

            else:

                device = torch.device("cpu")

                logger.warning(
                    "CUDA not available. Using CPU."
                )

            return device

        except Exception as e:

            raise MedicalDiagnosisException(e)

    @staticmethod
    def device_info():

        try:

            info = {

                "CUDA Available": torch.cuda.is_available(),

                "GPU Count": torch.cuda.device_count(),

                "Current Device": (
                    torch.cuda.current_device()
                    if torch.cuda.is_available()
                    else "CPU"
                ),

                "GPU Name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "No GPU"
                )

            }

            logger.info(info)

            return info

        except Exception as e:

            raise MedicalDiagnosisException(e)


if __name__ == "__main__":

    manager = DeviceManager()

    device = manager.get_device()

    print(f"\nSelected Device : {device}\n")

    info = manager.device_info()

    for key, value in info.items():

        print(f"{key}: {value}")