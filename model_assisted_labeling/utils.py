"""Module with used function for Labelbox.com"""

import numpy as np

from skimage.measure import find_contours
from matplotlib import patches
    
def extract_annotations(
    boxes,
    masks,
    class_ids, 
    class_names,
    scores=None,
) -> list:
    """
    Function to extract the polygon annotations from Mask RCNN detections.

    Args:
        - boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        - masks: [height, width, num_instances]
        - class_ids: [num_instances]
        - class_names: list of class names of the dataset
        - scores: (optional) confidence scores for each box
    
    Returns:
        - annotations: list with polygons
    """

    annotations = []

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to extract *** \n")
        return None
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]


    for i in range(N):
        instance = {
            'name': '',
            'polygon': []
        }

        if not np.any(boxes[i]):
            continue

        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        caption = "{} {:.3f}".format(label, score) if score else label

        instance['name'] = label

        mask = masks[:, :, i]
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)

        for verts in contours:
            verts = np.fliplr(verts) - 1
            p = patches.Polygon(verts, facecolor="none")

            p = p.get_xy()
            p = [{'x': i[0], 'y': i[1]} for i in p]
            instance['polygon'] += p

        annotations.append(instance)
    
    return annotations
