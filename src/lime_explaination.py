import numpy as np
import torch

from PIL import Image

from lime import lime_image

from skimage.segmentation import mark_boundaries


def create_predict_function(model, transform, device):
    """
    Returns a prediction function compatible with LIME.
    """

    def predict(images):

        batch = []

        for img in images:

            pil_img = Image.fromarray(
                img.astype(np.uint8)
            )

            tensor = transform(
                pil_img
            )

            batch.append(tensor)

        batch = torch.stack(batch).to(device)

        with torch.no_grad():

            outputs = model(batch)

            probs = torch.softmax(
                outputs,
                dim=1
            )

        return probs.cpu().numpy()

    return predict


def generate_lime(
        image_pil,
        predict_function,
        num_samples=1000,
        num_features=15
):

    image_np = np.array(
        image_pil.convert("RGB")
    )

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image_np,
        predict_function,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )

    return mark_boundaries(
        temp / 255.0,
        mask
    )