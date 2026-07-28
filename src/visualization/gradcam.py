"""
==========================================================
Medical Diagnosis AI
Grad-CAM Visualization
==========================================================

Author : Kinza Arshad

Description:
Generate Explainable AI heatmaps using Grad-CAM.
==========================================================
"""

"""
==========================================================
Medical Diagnosis AI
Grad-CAM Visualization
==========================================================

Author : Kinza Arshad

Description:
Generate Explainable AI heatmaps using Grad-CAM.
==========================================================
"""


import cv2
import numpy as np
import torch
from pathlib import Path


from src.models.model import MedicalDiagnosisModel
from src.utils.device import DeviceManager
from src.utils.checkpoint import CheckpointManager
from src.data.augmentation import ImageAugmentation

from src.utils.logger import logger
from src.utils.custom_exception import MedicalDiagnosisException



class GradCAM:


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

            # Enable gradients for Grad-CAM
            self.model.requires_grad_(True)


            self.transform = (
                ImageAugmentation()
                .test_augmentation()
            )


            self.activations = None
            self.gradients = None


            self.target_layer = (
                self.get_target_layer()
            )


            self.register_hooks()


            logger.info(
                "GradCAM initialized."
            )


        except Exception as e:

            raise MedicalDiagnosisException(e)



    # --------------------------------------------------
    # Target Layer
    # --------------------------------------------------

    def get_target_layer(self):


        model = self.model


        # ResNet50

        if hasattr(model, "layer4"):

            logger.info(
                "GradCAM target: ResNet layer4[-1]"
            )

            return model.layer4[-1]



        # DenseNet121

        if hasattr(model, "features"):

            logger.info(
                "GradCAM target: DenseNet features[-1]"
            )

            return model.features[-1]



        raise ValueError(
            "Unsupported model architecture"
        )



    # --------------------------------------------------
    # Hooks
    # --------------------------------------------------

    def register_hooks(self):


        def forward_hook(
            module,
            input,
            output
        ):

            self.activations = output



        def backward_hook(
            module,
            grad_input,
            grad_output
        ):

            self.gradients = grad_output[0]



        self.target_layer.register_forward_hook(
            forward_hook
        )


        self.target_layer.register_full_backward_hook(
            backward_hook
        )


        logger.info(
            "GradCAM hooks registered."
        )



    # --------------------------------------------------
    # Preprocess
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
                f"Image not found: {image_path}"
            )


        original_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        transformed = self.transform(
            image=original_image
        )


        tensor = (
            transformed["image"]
            .unsqueeze(0)
            .to(self.device)
        )


        return tensor, original_image



    # --------------------------------------------------
    # Generate Heatmap
    # --------------------------------------------------

    def generate_heatmap(
        self,
        image_path
    ):


        try:


            self.model.eval()

            self.model.requires_grad_(True)


            self.activations = None

            self.gradients = None



            input_tensor, original_image = (
                self.preprocess_image(
                    image_path
                )
            )



            self.model.zero_grad()



            output = self.model(
                input_tensor
            )



            predicted_class = torch.argmax(
                output,
                dim=1
            )



            score = output[
                0,
                predicted_class.item()
            ]



            score.backward()



            if self.activations is None:

                raise RuntimeError(
                    "Activation hook failed"
                )



            if self.gradients is None:

                raise RuntimeError(
                    "Gradient hook failed"
                )



            gradients = self.gradients[0]

            activations = self.activations[0]



            weights = torch.mean(
                gradients,
                dim=(1,2)
            )



            cam = torch.zeros(
                activations.shape[1:],
                device=self.device
            )



            for index, weight in enumerate(weights):

                cam += (
                    weight *
                    activations[index]
                )



            cam = torch.relu(cam)



            cam -= cam.min()


            cam /= (
                cam.max()
                +
                1e-8
            )



            cam = (
                cam.detach()
                .cpu()
                .numpy()
            )



            cam = cv2.resize(
                cam,
                (
                    original_image.shape[1],
                    original_image.shape[0]
                )
            )



            return (
                cam,
                original_image,
                predicted_class.item()
            )



        except Exception as e:

            raise MedicalDiagnosisException(e)



    # --------------------------------------------------
    # Overlay
    # --------------------------------------------------

    def overlay_heatmap(
        self,
        image_path,
        alpha=0.4,
        output_path="outputs/gradcam/gradcam_result.png"
    ):


        try:


            cam, image, predicted_class = (
                self.generate_heatmap(
                    image_path
                )
            )



            heatmap = np.uint8(
                255 * cam
            )


            heatmap = cv2.applyColorMap(
                heatmap,
                cv2.COLORMAP_JET
            )



            image_bgr = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )



            overlay = cv2.addWeighted(
                image_bgr,
                1-alpha,
                heatmap,
                alpha,
                0
            )



            Path(output_path).parent.mkdir(
                parents=True,
                exist_ok=True
            )



            cv2.imwrite(
                output_path,
                overlay
            )



            return {

                "predicted_class": predicted_class,

                "heatmap": heatmap,

                "overlay": overlay,

                "output_path": output_path

            }



        except Exception as e:

            raise MedicalDiagnosisException(e)
