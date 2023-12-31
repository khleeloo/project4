import cv2
import numpy as np
from loguru import logger

from superpoint_superglue_deployment import Matcher


def ImageMatch(img1, img2):


    superglue_matcher = Matcher(
        {
            "superpoint": {
                "input_shape": (-1, -1),
                "keypoint_threshold": 0.003,
            },
            "superglue": {
                "match_threshold": 0.5,
            },
            "use_gpu": True,
        }
    )
    query_kpts, ref_kpts, _, _, matches = superglue_matcher.match(img1, img2)
    kp1=np.float64([query_kpts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    kp2=np.float64([ref_kpts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    #Only matching 3< datapints
    
    if kp1.shape[0]<=3:
        method=None
        return kp1,kp2,matches,method,_
    M, mask = cv2.findHomography(
        kp1,
        kp2,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=5.0,
        maxIters=10000,
        confidence=0.95,
    )
    logger.info(f"number of inliers: {mask.sum()}")

    matches = np.array(matches)[np.all(mask > 0, axis=1)]
    matches = sorted(matches[:50], key=lambda match: match.distance)
    matched_image = cv2.drawMatches(
        img1,
        query_kpts,
        img2,
        ref_kpts,
        matches,
        None,
        flags=2,
    )
    # cv2.imshow('show all matchings', matched_image)
    # cv2.waitKey()
    # cv2.imwrite("matched_image.jpg", matched_image)
    # kp1=np.float64([query_kpts[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
    # kp2=np.float64([ref_kpts[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 1, 2)
    n_inliners=mask.sum()
    if n_inliners<7:
        method=None
    elif n_inliners==7:
        method='7_point'
    else:
        method='RANSAC'
    return kp1,kp2,matches,method, n_inliners