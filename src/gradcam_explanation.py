import torch
import numpy as np
import cv2


def generate_gradcam(model, image_pil, transform, device):
    """
    Generate Grad-CAM heatmap for any ResNet model.

    Args:
        model: PyTorch model
        image_pil: PIL Image
        transform: torchvision transform
        device: cpu/cuda

    Returns:
        RGB image with Grad-CAM overlay
    """

    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    image_tensor.requires_grad_(True)

    gradients = None
    activations = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    target_layer = model.layer4[-1]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)

    pred_class = output.argmax(dim=1).item()

    model.zero_grad()

    output[:, pred_class].backward()

    pooled_gradients = torch.mean(
        gradients,
        dim=[0, 2, 3]
    )

    activations = activations.squeeze(0)

    activations *= pooled_gradients.view(-1, 1, 1)

    heatmap = torch.mean(
        activations,
        dim=0
    ).cpu().detach().numpy()

    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    img = np.array(
        image_pil.resize((224, 224))
    )

    heatmap = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    output_img = heatmap * 0.4 + img

    forward_handle.remove()
    backward_handle.remove()

    return output_img.astype(np.uint8)