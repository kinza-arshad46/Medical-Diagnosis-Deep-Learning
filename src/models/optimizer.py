"""
==========================================================
Medical Diagnosis AI
Optimizer Module
==========================================================

Author : Kinza Arshad

Description:
Creates configurable optimizers for model training.
Supports Adam, AdamW, SGD and RMSprop.
==========================================================
"""

import torch.optim as optim

from src.utils.config_loader import ConfigLoader
from src.utils.logger import logger
from src.utils.custom_exception import MedicalDiagnosisException


class OptimizerBuilder:

    def __init__(self):

        self.config = ConfigLoader()

        self.optimizer_name = self.config.get(
            "optimizer",
            "name"
        )

        self.learning_rate = self.config.get(
            "training",
            "learning_rate"
        )

        self.weight_decay = self.config.get(
            "optimizer",
            "weight_decay",
            default=0.0
        )

        self.momentum = self.config.get(
            "optimizer",
            "momentum",
            default=0.9
        )

    def get_optimizer(self, model):

        """
        Returns configured optimizer.
        """

        try:

            trainable_parameters = filter(
                lambda p: p.requires_grad,
                model.parameters()
            )

            optimizer_name = self.optimizer_name.lower()

            if optimizer_name == "adam":

                optimizer = optim.Adam(
                    trainable_parameters,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                )

            elif optimizer_name == "adamw":

                optimizer = optim.AdamW(
                    trainable_parameters,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                )

            elif optimizer_name == "sgd":

                optimizer = optim.SGD(
                    trainable_parameters,
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay
                )

            elif optimizer_name == "rmsprop":

                optimizer = optim.RMSprop(
                    trainable_parameters,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    momentum=self.momentum
                )

            else:

                raise ValueError(
                    f"Unsupported optimizer: {self.optimizer_name}"
                )

            logger.info(
                f"{self.optimizer_name} optimizer initialized."
            )

            return optimizer

        except Exception as e:

            raise MedicalDiagnosisException(e)


if __name__ == "__main__":

    from src.models.model import MedicalDiagnosisModel

    model = MedicalDiagnosisModel().get_model()

    optimizer = OptimizerBuilder().get_optimizer(model)

    print(optimizer)