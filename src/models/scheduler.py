"""
==========================================================
Medical Diagnosis AI
Learning Rate Scheduler
==========================================================

Author : Kinza Arshad

Description:
Creates configurable learning rate schedulers
for model training.
==========================================================
"""

import torch.optim.lr_scheduler as lr_scheduler

from src.utils.config_loader import ConfigLoader
from src.utils.logger import logger
from src.utils.custom_exception import MedicalDiagnosisException


class SchedulerBuilder:

    def __init__(self):

        self.config = ConfigLoader()

        self.scheduler_name = self.config.get(
            "scheduler",
            "name",
            default="StepLR"
        )

        self.step_size = self.config.get(
            "scheduler",
            "step_size",
            default=5
        )

        self.gamma = self.config.get(
            "scheduler",
            "gamma",
            default=0.1
        )

        self.patience = self.config.get(
            "scheduler",
            "patience",
            default=3
        )

        self.factor = self.config.get(
            "scheduler",
            "factor",
            default=0.1
        )

        self.t_max = self.config.get(
            "scheduler",
            "t_max",
            default=20
        )

    def get_scheduler(self, optimizer):

        try:

            scheduler_type = self.scheduler_name.lower()

            if scheduler_type == "steplr":

                scheduler = lr_scheduler.StepLR(
                    optimizer,
                    step_size=self.step_size,
                    gamma=self.gamma
                )

            elif scheduler_type == "reducelronplateau":

                scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.factor,
                    patience=self.patience
                )

            elif scheduler_type == "cosineannealinglr":

                scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.t_max
                )

            elif scheduler_type == "exponentiallr":

                scheduler = lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=self.gamma
                )

            else:

                raise ValueError(
                    f"Unsupported Scheduler: {self.scheduler_name}"
                )

            logger.info(
                f"{self.scheduler_name} initialized successfully."
            )

            return scheduler

        except Exception as e:

            raise MedicalDiagnosisException(e)


if __name__ == "__main__":

    from src.models.model import MedicalDiagnosisModel
    from src.models.optimizer import OptimizerBuilder

    model = MedicalDiagnosisModel().get_model()

    optimizer = OptimizerBuilder().get_optimizer(model)

    scheduler = SchedulerBuilder().get_scheduler(
        optimizer
    )

    print(scheduler)