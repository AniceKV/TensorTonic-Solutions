def iou(boxa, boxb):
    x_left = max(boxa[0], boxb[0])
    y_top = max(boxa[1], boxb[1])
    x_right = min(boxa[2], boxb[2])
    y_bottom = min(boxa[3], boxb[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)

    area_a = (boxa[2] - boxa[0]) * (boxa[3] - boxa[1])
    area_b = (boxb[2] - boxb[0]) * (boxb[3] - boxb[1])

    union = area_a + area_b - intersection

    return intersection / union