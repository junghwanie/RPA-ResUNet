import numpy as np


def iou(pred, target, n_classes=2):

    iou = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for background class ("0")
    for cls in range(
        1, n_classes
    ):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (
            (pred_inds[target_inds]).long().sum().data.cpu().item()
        )  # Cast to long to prevent overflows
        union = (
            pred_inds.long().sum().data.cpu().item()
            + target_inds.long().sum().data.cpu().item()
            - intersection
        )

        if union == 0:
            iou.append(
                float("nan")
            )  # If there is no ground truth, do not include in evaluation
        else:
            iou.append(float(intersection) / float(max(union, 1)))

    return sum(iou)


def iou_metric(y_pred, y_true, n_classes=2):

    miou = []
    for i in np.arange(0.5, 1.0, 0.05):
        y_pred_ = y_pred > i
        iou_init = iou(y_pred_, y_true, n_classes=n_classes)
        miou.append(iou_init)

    return sum(miou) / len(miou)