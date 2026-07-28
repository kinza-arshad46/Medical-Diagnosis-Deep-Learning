"""
==========================================================
Medical Diagnosis AI
Loss Function
==========================================================

Author : Kinza Arshad

Description:
Creates configurable loss functions for
Medical Image Classification.
==========================================================
"""

import torch
import torch.nn as nn

from src.utils.config_loader import ConfigLoader
from src.utils.logger import logger
from src.utils.custom_exception import MedicalDiagnosisException


class LossFunction:

    def __init__(self):

        self.config = ConfigLoader()

    def get_loss(self, class_weights=None):

        """
        Returns CrossEntropyLoss.

        Parameters
        ----------
        class_weights : list or torch.Tensor, optional
            Class weights for imbalanced datasets.
        """

        try:

            weight_tensor = None

            if class_weights is not None:

                if not isinstance(class_weights, torch.Tensor):

                    weight_tensor = torch.tensor(
                        class_weights,
                        dtype=torch.float32
                    )

                else:

                    weight_tensor = class_weights.float()

            loss = nn.CrossEntropyLoss(
                weight=weight_tensor,
                label_smoothing=self.config.get(
                    "loss",
                    "label_smoothing",
                    default=0.0
                )
            )

            logger.info(
                "CrossEntropyLoss initialized successfully."
            )

            return loss

        except Exception as e:

            raise MedicalDiagnosisException(e)


if __name__ == "__main__":

    criterion = LossFunction().get_loss()

    print(criterion)