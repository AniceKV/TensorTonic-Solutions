import numpy as np
def generate_anchors(feature_size: int, image_size: float, scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    """
    Generate anchor boxes for object detection.
    """
    stride=image_size/feature_size

    centres=np.arange(feature_size)
    centres=0.5*stride + stride*centres 

    centres=[(i,j) for i in centres for j in centres]

    res=[]

    for x,y in centres:
        for side in scales:
            for ratio in aspect_ratios:
                width=side*np.sqrt(ratio)
                height=side/np.sqrt(ratio)
                
                res.append([
                y - width / 2,
                x - height / 2,
                y + width / 2,
                x + height / 2
            ])
    return res