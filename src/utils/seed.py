"""
==========================================================
Medical Diagnosis AI
Random Seed Utility
==========================================================

Author : Kinza Arshad

Description:
Sets random seeds for reproducible Deep Learning
experiments across Python, NumPy and PyTorch.
==========================================================
"""

import random
import numpy as np
import torch

from src.utils.logger import logger
from src.utils.custom_exception import MedicalDiagnosisException


class SeedManager:
    """
    Utility class to set random seeds.
    """

    @staticmethod
    def set_seed(seed: int = 42):

        try:

            random.seed(seed)

            np.random.seed(seed)

            torch.manual_seed(seed)

            if torch.cuda.is_available():

                torch.cuda.manual_seed(seed)

                torch.cuda.manual_seed_all(seed)

            torch.backends.cudnn.deterministic = True

            torch.backends.cudnn.benchmark = False

            logger.info(
                f"Random Seed set successfully: {seed}"
            )

        except Exception as e:

            raise MedicalDiagnosisException(e)


if __name__ == "__main__":

    SeedManager.set_seed(42)

    print("Random seed initialized successfully.")