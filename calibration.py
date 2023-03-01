import numpy as np
import cv2 
import glob, os


# Defining the dimensions of checkerboard
CHECKERBOARD = (9,12)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # termination criteria

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objpoints.append(objp)

# Find the checkerboard corners
# If it cannot find the corners properly, you can get only once the corners manually 
# But, you must do once again if the location of the camera is changed
ret = True
# Below is our case, so you need to change it into your own
corners = [ [[54.0,104.0]],
            [[41.0,232.0]],
            [[31.0,366.0]],
            [[25.0,503.0]],
            [[21.0,642.0]],
            [[18.0,785.0]],
            [[19.0,923.0]],
            [[23.0,1061.0]],
            [[28.0,1198.0]],
            
            [[177.0,96.0]],
            [[167.0,228.0]],
            [[158.0,363.0]],
            [[151.0,502.0]],
            [[147.0,645.0]],
            [[145.0,787.0]],
            [[146.0,930.0]],
            [[149.0,1071.0]],
            [[154.0,1208.0]],
            
            [[307.0,88.0]],
            [[298.0,218.0]],
            [[291.0,357.0]],
            [[285.0,500.0]],
            [[282.0,645.0]],
            [[280.0,792.0]],
            [[280.0,937.0]],
            [[282.0,1079.0]],
            [[288.0,1218.0]],
            
            [[442.0,82.0]],
            [[435.0,216.0]],
            [[429.0,355.0]],
            [[425.0,499.0]],
            [[423.0,647.0]],
            [[421.0,794.0]],
            [[421.0,940.0]],
            [[423.0,1087.0]],
            [[425.0,1226.0]],
            
            [[581.0,77.0]],
            [[577.0,213.0]],
            [[574.0,355.0]],
            [[570.0,497.0]],
            [[568.0,646.0]],
            [[566.0,797.0]],
            [[566.0,945.0]],
            [[568.0,1091.0]],
            [[568.0,1233.0]],
            
            [[723.0,76.0]],
            [[722.0,210.0]],
            [[719.0,353.0]],
            [[718.0,497.0]],
            [[717.0,649.0]],
            [[716.0,798.0]],
            [[716.0,947.0]],
            [[715.0,1094.0]],
            [[715.0,1237.0]],
            
            [[866.0,77.0]],
            [[867.0,212.0]],
            [[867.0,353.0]],
            [[868.0,497.0]],
            [[867.0,647.0]],
            [[867.0,797.0]],
            [[866.0,948.0]],
            [[864.0,1095.0]],
            [[863.0,1240.0]],
            
            [[1010.0,78.0]],
            [[1012.0,214.0]],
            [[1014.0,355.0]],
            [[1016.0,499.0]],
            [[1016.0,649.0]],
            [[1015.0,797.0]],
            [[1015.0,945.0]],
            [[1013.0,1095.0]],
            [[1010.0,1235.0]],
            
            [[1148.0,84.0]],
            [[1154.0,220.0]],
            [[1158.0,357.0]],
            [[1160.0,503.0]],
            [[1163.0,649.0]],
            [[1161.0,797.0]],
            [[1161.0,944.0]],
            [[1159.0,1090.0]],
            [[1154.0,1231.0]],
            
            [[1286.0,91.0]],
            [[1291.0,224.0]],
            [[1297.0,363.0]],
            [[1301.0,504.0]],
            [[1303.0,650.0]],
            [[1303.0,796.0]],
            [[1302.0,941.0]],
            [[1299.0,1086.0]],
            [[1294.0,1226.0]],
            
            [[1414.0,100.0]],
            [[1423.0,229.0]],
            [[1430.0,367.0]],
            [[1436.0,506.0]],
            [[1439.0,649.0]],
            [[1439.0,793.0]],
            [[1437.0,939.0]],
            [[1435.0,1079.0]],
            [[1427.0,1215.0]],
            
            [[1539.0,113.0]],
            [[1550.0,239.0]],
            [[1557.0,375.0]],
            [[1564.0,513.0]],
            [[1568.0,649.0]],
            [[1567.0,790.0]],
            [[1566.0,932.0]],
            [[1561.0,1071.0]],
            [[1553.0,1205.0]]
        ]

# corners = sorted(corners, reverse=True)
corners = np.array(corners).astype(np.float32)
imgpoints.append(corners)

# print('corners : ', corners)

# Below is our case, so you need to change them into your own
input_dir = './assets/distorted_input/'
output_dir = './assets/undistorted_output/'

images = glob.glob(input_dir + '*.*')
images.sort()

for fname in images:
    
    print('fname:', fname, '\n')

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ## draw the points
    # for ptr in corners:
    #     print(ptr[0].astype(int))
    #     cv2.circle(gray, ptr[0].astype(int), 10, 0, -1)
    # cv2.namedWindow('draw_points', cv2.WINDOW_NORMAL)
    # cv2.imshow('draw_points', gray)
    # cv2.waitKey()
    
    # break

    # ret, corners = cv2.findChessboardCorners(gray, (7,12), None)

    # cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)

    """
        Performing camera calibration by 
        passing the value of known 3D points (objpoints)
        and corresponding pixel coordinates of the 
        detected corners (imgpoints)
    """
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # print("Camera matrix : \n")
    # print(mtx)
    # print("dist : \n")
    # print(dist)
    # print("rvecs : \n")
    # print(rvecs)
    # print("tvecs : \n")
    # print(tvecs)

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    
    name = os.path.basename(fname)  
    out_fname = output_dir + name
    # save the result
    cv2.imwrite(out_fname, dst)
