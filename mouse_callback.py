import cv2


def mouse_event(event, x, y, flags, param):
    
    if event == cv2.EVENT_FLAG_LBUTTON:    
        cv2.circle(param, (x, y), 3, (255, 0, 0), 2)
        cv2.imshow("draw", img)
        
        print('x,y : ', x, y)
        f.write('[['+str(x)+','+str(y)+']]' + ',\n')

f = open('assets/corners.txt', 'w')

# img = cv2.imread('assets/checkerboard/checkerboard.jpg')
img = cv2.imread('assets/distorted_input/1.jpg')
cv2.imshow("draw", img)
cv2.setMouseCallback("draw", mouse_event, img)
cv2.waitKey()

f.close()