import cv2
import torch
from ultralytics.data.augment import LetterBox


def preprocess_image(img, new_shape=(640, 640)):
    """
    process input image for model, output a image tensor
    """
    # letterbox
    letterbox = LetterBox(
        new_shape=new_shape, auto=False, scaleFill=False, scaleup=False
    )
    letterbox_img = letterbox(image=img)

    # convert to rgb
    rgb_img = cv2.cvtColor(letterbox_img, cv2.COLOR_BGR2RGB)

    # denoise
    # rgb_img = cv2.fastNlMeansDenoisingColored(rgb_img, None, 10, 10, 7, 21)
    denoised_img = cv2.GaussianBlur(rgb_img, (5, 5), 0)

    # normalize to [0, 1]
    img_normalized = (
        torch.from_numpy(denoised_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    )

    return img_normalized
