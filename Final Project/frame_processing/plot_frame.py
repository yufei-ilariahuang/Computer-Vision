import cv2


def plot_bbox(img, bbox):
    """
    plot bbox on image

    Args:
        img: image to plot bbox on
        bbox: bbox to plot, numpy array [x1, y1, x2, y2]
    """
    pt1 = (int(bbox[0]), int(bbox[1]))
    pt2 = (int(bbox[2]), int(bbox[3]))
    cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)
    return img


def draw_text_background(img, text, position, color, font_size=1, thickness=1):
    """
    Draw background for text

    Args:
        img: image to draw on
        text: text to draw
        position: position to draw text (x, y)
        font_scale: scale of the font
        thickness: thickness of the text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = 5
    text_size = cv2.getTextSize(text, font, font_size, thickness)[0]
    background_topleft = (position[0], position[1] - text_size[1] - padding)
    background_bottomright = (position[0] + text_size[0], position[1] + padding)
    cv2.rectangle(img, background_topleft, background_bottomright, color, cv2.FILLED)
    return img


def plot_class_confidence(
    img, class_name, confidence, bbox, background_color, font_size=1, thickness=1
):
    """
    Plot class and confidence above the bbox

    Args:
        img: image to plot on
        class_name: name of the class
        confidence: confidence level
        bbox: bbox to plot above, numpy array [x1, y1, x2, y2]
    """
    text = f"{class_name}: {confidence:.2f}"
    position = (int(bbox[0]), int(bbox[1]) - 40)
    img = draw_text_background(
        img, text, position, background_color, font_size, thickness
    )
    cv2.putText(
        img,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        thickness,
    )
    return img


def plot_distance(img, distance, bbox, background_color, font_size=1, thickness=1):
    """
    Plot distance below the class and confidence

    Args:
        img: image to plot on
        distance: distance to plot, in meters
        bbox: bbox to plot distance on, numpy array [x1, y1, x2, y2]
    """
    if distance is None:
        text = f"Distance: N/A"
    else:
        text = f"Distance: {distance:.2f}m"
    position = (int(bbox[0]), int(bbox[1]) - 10)
    img = draw_text_background(
        img, text, position, background_color, font_size, thickness
    )
    cv2.putText(
        img,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        thickness,
    )
    return img
