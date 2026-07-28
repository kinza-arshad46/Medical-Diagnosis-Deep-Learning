"""
==========================================================
Medical Diagnosis AI
Transfer Learning Model Builder
==========================================================

Author : Kinza Arshad

Description:
Production-ready Transfer Learning Models
with Fine-Tuning support.
==========================================================
"""

import torch.nn as nn
from torchvision import models

from src.utils.config_loader import ConfigLoader
from src.utils.logger import logger
from src.utils.custom_exception import MedicalDiagnosisException


class MedicalDiagnosisModel:

    def __init__(self):

        self.config = ConfigLoader()

        self.model_name = self.config.get("model", "name")

        self.pretrained = self.config.get(
            "model",
            "pretrained",
            default=False
        )

        self.num_classes = self.config.get(
            "model",
            "num_classes"
        )

        self.dropout = self.config.get(
            "model",
            "dropout"
        )

        self.freeze_backbone = self.config.get(
            "model",
            "freeze_backbone",
            default=True
        )

        self.unfreeze_layers = self.config.get(
            "model",
            "unfreeze_layers",
            default=2
        )

    def freeze_model(self, model):

        for parameter in model.parameters():
            parameter.requires_grad = False

    def unfreeze_last_layers(self, model):

        children = list(model.children())

        for child in children[-self.unfreeze_layers:]:
            for parameter in child.parameters():
                parameter.requires_grad = True

    def build_resnet50(self):

        try:

            # Never download weights
            model = models.resnet50(weights=None)

            if self.freeze_backbone:
                self.freeze_model(model)
                self.unfreeze_last_layers(model)

            in_features = model.fc.in_features

            model.fc = nn.Sequential(

                nn.Linear(in_features, 512),

                nn.BatchNorm1d(512),

                nn.ReLU(inplace=True),

                nn.Dropout(self.dropout),

                nn.Linear(512, 256),

                nn.ReLU(inplace=True),

                nn.Dropout(0.30),

                nn.Linear(256, self.num_classes)

            )

            logger.info("ResNet50 created successfully.")

            return model

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def build_densenet121(self):

        try:

            # Never download weights
            model = models.densenet121(weights=None)

            if self.freeze_backbone:
                self.freeze_model(model)
                self.unfreeze_last_layers(model)

            in_features = model.classifier.in_features

            model.classifier = nn.Sequential(

                nn.Linear(in_features, 512),

                nn.BatchNorm1d(512),

                nn.ReLU(inplace=True),

                nn.Dropout(self.dropout),

                nn.Linear(512, 256),

                nn.ReLU(inplace=True),

                nn.Dropout(0.30),

                nn.Linear(256, self.num_classes)

            )

            logger.info("DenseNet121 created successfully.")

            return model

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def get_model(self):

        if self.model_name.lower() == "resnet50":
            return self.build_resnet50()

        elif self.model_name.lower() == "densenet121":
            return self.build_densenet121()

        else:
            raise ValueError(
                f"Unsupported model: {self.model_name}"
            )


if __name__ == "__main__":

    builder = MedicalDiagnosisModel()

    model = builder.get_model()

    print(model)
