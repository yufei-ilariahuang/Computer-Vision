def convert_bbox_to_origin_img(bboxes, original_shape, pred_shape):
    """
    convert bbox in pred image to original image

    bboxes: numpy array [x1, y1, x2, y2]
    original_shape: shape of original image before letterbox
    pred_shape: shape of image input to model
    """
    scale_ratio = min(
        pred_shape[0] / original_shape[0], pred_shape[1] / original_shape[1]
    )
    padding = (pred_shape[1] - original_shape[1] * scale_ratio) / 2, (
        pred_shape[0] - original_shape[0] * scale_ratio
    ) / 2

    bboxes[:, [0, 2]] -= padding[0]  # remove horizontal padding
    bboxes[:, [1, 3]] -= padding[1]  # remove vertical padding
    bboxes[:, :4] /= scale_ratio  # scale bbox to original image

    # clip bbox to original image
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, original_shape[1])
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, original_shape[0])

    return bboxes
