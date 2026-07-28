"""
==========================================================
Medical Diagnosis AI
Prediction Module
==========================================================

Author : Kinza Arshad

Description:
Single image prediction and confidence generation.
==========================================================
"""


import time
import torch
import cv2
import numpy as np


from src.models.model import MedicalDiagnosisModel
from src.utils.device import DeviceManager
from src.utils.checkpoint import CheckpointManager
from src.data.augmentation import ImageAugmentation

from src.utils.logger import logger
from src.utils.custom_exception import MedicalDiagnosisException



class Predictor:


    def __init__(self):

        try:

            self.device = DeviceManager.get_device()


            self.model = (
                MedicalDiagnosisModel()
                .get_model()
                .to(self.device)
            )


            CheckpointManager().load_checkpoint(
                model=self.model,
                best=True
            )


            self.model.eval()


            self.transform = (
                ImageAugmentation()
                .test_augmentation()
            )


            self.classes = [
                "Normal",
                "Pneumonia"
            ]


            logger.info(
                "Predictor initialized successfully."
            )


        except Exception as e:

            raise MedicalDiagnosisException(e)



    # --------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------

    def preprocess_image(
        self,
        image_path
    ):


        image = cv2.imread(
            str(image_path)
        )


        if image is None:

            raise FileNotFoundError(
                f"Unable to read image {image_path}"
            )



        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        transformed = self.transform(
            image=image
        )


        tensor = (
            transformed["image"]
            .unsqueeze(0)
            .to(self.device)
        )


        return tensor



    # --------------------------------------------------
    # Single Prediction
    # --------------------------------------------------

    def predict_single(
        self,
        image_path
    ):


        try:


            start_time = time.time()



            image_tensor = (
                self.preprocess_image(
                    image_path
                )
            )



            # IMPORTANT:
            # GradCAM needs gradients,
            # therefore do not use torch.no_grad here.

            with torch.enable_grad():


                output = self.model(
                    image_tensor
                )



                probabilities = torch.softmax(
                    output,
                    dim=1
                )



                confidence, index = torch.max(
                    probabilities,
                    dim=1
                )



            prediction_time = (
                time.time()
                -
                start_time
            )



            result = {


                "predicted_class":
                    self.classes[
                        index.item()
                    ],


                "class_index":
                    index.item(),


                "confidence":
                    confidence.item(),


                "prediction_time":
                    prediction_time

            }



            return result



        except Exception as e:

            raise MedicalDiagnosisException(e)



    # --------------------------------------------------
    # Top K Predictions
    # --------------------------------------------------

    def predict_top_k(
        self,
        image_path,
        top_k=2
    ):


        try:


            image_tensor = (
                self.preprocess_image(
                    image_path
                )
            )



            with torch.enable_grad():


                output = self.model(
                    image_tensor
                )



                probabilities = torch.softmax(
                    output,
                    dim=1
                )



            values, indices = torch.topk(
                probabilities,
                top_k
            )



            results = []



            for value, index in zip(
                values[0],
                indices[0]
            ):


                results.append(

                    {

                    "class":
                        self.classes[
                            index.item()
                        ],

                    "probability":
                        value.item()

                    }

                )



            return results



        except Exception as e:

            raise MedicalDiagnosisException(e)
