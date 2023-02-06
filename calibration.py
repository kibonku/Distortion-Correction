import numpy as np
import cv2 
import glob, os


# Defining the dimensions of checkerboard
CHECKERBOARD = (7, 12)
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
corners = [[[5893., 401.]],
           [[5889., 945.]],
           [[5885, 1481]],
           [[5881., 2001.]],
           [[5869., 2535.]],
           [[5853., 3047.]],
           [[5837., 3555.]],

           [[5369., 401.]],
           [[5377., 945.]],
           [[5381., 1485.]],
           [[5377., 2019.]],
           [[5369., 2547.]],
           [[5353., 3063.]],
           [[5329., 3571.]],

           [[4849., 401.]],
           [[4861., 941.]],
           [[4865., 1489.]],
           [[4861., 2027.]],
           [[4857., 2563.]],
           [[4845., 3083.]],
           [[4829., 3587.]],

           [[4313., 397.]],
           [[4325., 941.]],
           [[4337., 1489.]],
           [[4341., 2040.]],
           [[4333., 2580.]],
           [[4329., 3104.]],
           [[4313., 3608.]],

           [[3769., 393.]],
           [[3785., 941.]],
           [[3793., 1493.]],
           [[3801., 2044.]],
           [[3801., 2588.]],
           [[3801., 3116.]],
           [[3793., 3624.]],

           [[3225., 397.]],
           [[3237., 945.]],
           [[3245., 1501.]],
           [[3253., 2056.]],
           [[3257., 2600.]],
           [[3265., 3124.]],
           [[3269., 3636.]],

           [[2681., 397.]],
           [[2685., 961.]],
           [[2693., 1509.]],
           [[2701., 2064.]],
           [[2709., 2604.]],
           [[2725., 3132.]],
           [[2737., 3644.]],

           [[2137., 421.]],
           [[2141., 973.]],
           [[2145., 1517.]],
           [[2161., 2068.]],
           [[2177., 2604.]],
           [[2193., 3132.]],
           [[2213., 3644.]],

           [[1597., 449.]],
           [[1601., 981.]],
           [[1613., 1529.]],
           [[1625., 2076.]],
           [[1645., 2608.]],
           [[1669., 3132.]],
           [[1697., 3640.]],

           [[1077., 469.]],
           [[1081., 1005.]],
           [[1093., 1545.]],
           [[1109., 2080.]],
           [[1129., 2612.]],
           [[1161., 3128.]],
           [[1189., 3628.]],

           [[565., 493.]],
           [[565., 1017.]],
           [[585., 1553.]],
           [[605., 2085.]],
           [[633., 2605.]],
           [[657., 3121.]],
           [[689., 3624.]],

           [[57., 509.]],
           [[69., 1033.]],
           [[93., 1566.]],
           [[109., 2090.]],
           [[137., 2610.]],
           [[165., 3118.]],
           [[197., 3614.]]
           ]
corners = np.array(corners).astype(np.float32)
imgpoints.append(corners)

# Below is our case, so you need to change them into your own
input_dir = './assets/distorted_input/'
output_dir = './assets/undistorted_output/'

images = glob.glob(input_dir + '*.jpg')
images.sort()

for fname in images:
    
    print('fname:', fname, '\n')

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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
