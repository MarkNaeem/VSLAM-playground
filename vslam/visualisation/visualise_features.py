import cv2
import numpy as np
import torch
from datetime import datetime
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint

from vslam.utils import *
from vslam.definitions import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Running on device "{}"'.format(device))

img1 = cv2.imread(f'{DATASET_PATH}/sequences/02/image_2/000000.png', 0)
img2 = cv2.imread(f'{DATASET_PATH}/sequences/02/image_2/000001.png', 0)
print("image size is",img1.shape)
#img1 = scale_image(img1,max_height=256)
#img2 = scale_image(img2,max_height=256)
print("scaled image size is",img1.shape)

frame1 = frame2tensor(img1,device)
frame2 = frame2tensor(img2,device)

model = SuperPoint({'nms_radius': 4,'keypoint_threshold': 0.005,'max_keypoints': -1}).eval().to(device)

output = model({'image':frame1})
frame1_data = {k+'0': v for k, v in output.items()}
frame1_data['image0'] = frame1

st = datetime.now()
output = model({'image':frame2})
frame2_data = {k+'1': v for k, v in output.items()}
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(frame1_data['descriptors0'][0].cpu().detach().numpy().T, frame2_data['descriptors1'][0].cpu().detach().numpy().T, 2)
print("superpoint took",(datetime.now()-st))

#-- Filter matches using the Lowe's ratio test
ratio_thresh = 0.65
mkpts0 = []
mkpts1 = []
for m,n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        # Query index of the match in the previous image
        idx0 = m.queryIdx
        # Train index of the match in the current image
        idx1 = m.trainIdx

        # Get the coordinates of the matched keypoints
        pt0 = frame1_data['keypoints0'][0].cpu().detach().numpy()[idx0].astype(np.int32)
        pt1 = frame2_data['keypoints1'][0].cpu().detach().numpy()[idx1].astype(np.int32)
        mkpts0.append([pt0[0],pt0[1]])
        mkpts1.append([pt1[0],pt1[1]])            
        
#-- Draw matches
mtchs = []
for pt1, pt2 in zip(mkpts0,mkpts1):
    mtchs.append([pt1,pt2])
draw_flow(img2,mtchs)



# Configuration for the matching model
config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1
    },
    'superglue': {
        'weights': 'outdoor',  # or 'indoor'
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}
matching = Matching(config).eval().to(device)

st = datetime.now()
output = model({'image':frame2})
frame2_data = {k+'1': v for k, v in output.items()}
frame2_data['image1'] = frame2
# Perform matching
pred = matching({**frame1_data,**frame2_data})

kpts0 = frame1_data['keypoints0'][0].cpu().numpy()
kpts1 = frame2_data['keypoints1'][0].cpu().numpy()
matches = pred['matches0'][0].cpu().numpy()
valid = matches > -1
mkpts0 = kpts0[valid]
mkpts1 = kpts1[matches[valid]]
print("superglue took",(datetime.now()-st))

#-- Draw matches
mtchs = []
for pt1, pt2 in zip(mkpts0,mkpts1):
    mtchs.append([pt1,pt2])
draw_flow(img2,mtchs)



# Initialize sift
sift = cv2.SIFT_create()
keypoints_sift1, descriptors_sift1 = sift.detectAndCompute(img1, None)

st = datetime.now()
# Detect keypoints and descriptors with sift
keypoints_sift2, descriptors_sift2 = sift.detectAndCompute(img2, None)

# FLANN parameters and matcher for sift
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches_sift = flann.knnMatch(descriptors_sift1, descriptors_sift2, k=2)
print("sift took",(datetime.now()-st))

#-- Filter matches using the Lowe's ratio test
ratio_thresh = 0.65
mkpts0 = []
mkpts1 = []
for m,n in matches_sift:
    if m.distance < ratio_thresh * n.distance:
        # Query index of the match in the previous image
        idx0 = m.queryIdx
        # Train index of the match in the current image
        idx1 = m.trainIdx

        # Get the coordinates of the matched keypoints
        pt0 = keypoints_sift1[idx0]
        pt1 = keypoints_sift2[idx1]
        mkpts0.append([pt0.pt[0], pt0.pt[1]])
        mkpts1.append([pt1.pt[0], pt1.pt[1]])       

#-- Draw matches
mtchs = []
for pt1, pt2 in zip(mkpts0,mkpts1):
    mtchs.append([pt1,pt2])
draw_flow(img2,mtchs)



# Initialize ORB
orb = cv2.ORB_create()
keypoints_orb1 = cv2.goodFeaturesToTrack(img1, 3000, qualityLevel=0.01, minDistance=7)
keypoints_orb1 = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in keypoints_orb1]
keypoints_orb1, descriptors_orb1 = orb.compute(img1, keypoints_orb1)

st = datetime.now()
# Detect keypoints and descriptors with ORB
keypoints_orb2 = cv2.goodFeaturesToTrack(img2, 3000, qualityLevel=0.01, minDistance=7)
keypoints_orb2 = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in keypoints_orb2]
keypoints_orb2, descriptors_orb2 = orb.compute(img2, keypoints_orb2)

# BFMatcher with default params for ORB
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_orb = bf.match(descriptors_orb1, descriptors_orb2)
# Sort them in the order of their distance for ORB
ORB_DISTANCE_THRESHOLD = 60
matches_orb =  [x for x in matches_orb if x.distance<ORB_DISTANCE_THRESHOLD]
print("ORB took",(datetime.now()-st))

mkpts0 = []
mkpts1 = []
for m in matches_orb:
    # Query index of the match in the previous image
    idx0 = m.queryIdx
    # Train index of the match in the current image
    idx1 = m.trainIdx

    # Get the coordinates of the matched keypoints
    pt0 = keypoints_orb1[idx0].pt
    pt1 = keypoints_orb2[idx1].pt
    mkpts0.append([pt0[0], pt0[1]])
    mkpts1.append([pt1[0], pt1[1]])

#-- Draw matches
mtchs = []
for pt1, pt2 in zip(mkpts0,mkpts1):
    mtchs.append([pt1,pt2])
draw_flow(img2,mtchs)    
