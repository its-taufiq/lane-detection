# in this particular video we will try to detect the lanes for self driving car 

import numpy as np 
import cv2 
import matplotlib.pyplot as plt  


def get_canny(img, threshold1 = 100, threshold2 = 100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img, threshold1, threshold2)
    return canny 
    

def get_roi(edge, vertices):
    mask = np.zeros(edge.shape, np.uint8)
    color = 255
    cv2.fillPoly(mask, np.array([vertices], np.int32), color)
    result = cv2.bitwise_and(mask, edge)
    return result 



def create_lines(roi, frame, rho = 1, threshold = 100, min_length = 10, 
                 max_gap = 100 ):
    lines = cv2.HoughLinesP(roi, rho, np.pi/180, threshold, 
                            minLineLength = min_length, maxLineGap = max_gap) 
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)



def detect_objects(src1, src2):
    
    # gray = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
    
    # canny = cv2.Canny(src1, 100, 100)
    contours, _ = cv2.findContours(src1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < 300 or area>3000:
            pass 
        else:
            cv2.rectangle(src2, (x, y), (x+w, y+h), (0, 0, 255), 3)
            
    return src2 




def empty(x):
    pass 
 
    

def main():
    
    cap = cv2.VideoCapture('lane_video.mp4')
    
    window = 'tracker'
    cv2.namedWindow(window)
    cv2.resizeWindow(window, 500, 400)
    
    cv2.createTrackbar('save points', window, 0, 1, empty)
    cv2.createTrackbar('x1', window, 0, 700, empty)
    cv2.createTrackbar('y1', window, 0, 700, empty)
    
    cv2.createTrackbar('x2', window, 0, 700, empty)
    cv2.createTrackbar('y2', window, 0, 700, empty)
    
    cv2.createTrackbar('x3', window, 0, 700, empty)
    cv2.createTrackbar('y3', window, 0, 700, empty)
    
    cv2.createTrackbar('x4', window, 0, 700, empty)
    cv2.createTrackbar('y4', window, 0, 700, empty)
    
    
    # cv2.createTrackbar('rho', window, 1, 10, empty)
    # cv2.createTrackbar('threshold', window, 5, 500, empty)
    # cv2.createTrackbar('min length', window, 5, 200, empty)
    # cv2.createTrackbar('max gap', window, 5, 200, empty)
    # cv2.createTrackbar('save values', window, 0, 1, empty)
    
    i = 0
    while cap.isOpened():
        _, frame = cap.read()
        
        frame = cv2.resize(frame, (600, 400))
         
        if i == 0:
            plt.imshow(frame)
            i = i + 1
        
        x1 = cv2.getTrackbarPos('x1', window)
        y1 = cv2.getTrackbarPos('y1', window)
        
        x2 = cv2.getTrackbarPos('x2', window)
        y2 = cv2.getTrackbarPos('y2', window)
        
        x3 = cv2.getTrackbarPos('x3', window)
        y3 = cv2.getTrackbarPos('y3', window)
        
        x4 = cv2.getTrackbarPos('x4', window)
        y4 = cv2.getTrackbarPos('y4', window)

        save = cv2.getTrackbarPos('save points', window)
        
        if save == 1:
            
            points = [(x1, y1), (x2, y2), (x3 , y3), (x4, y4)]
            print(points)
         
        vertices = [(0, 440), (320, 230), (390, 230), (700, 400) ]

        canny = get_canny(frame)
        
        [(151, 325), (473, 322), (476, 154), (154, 154)]
        
        
        roi_objects = get_roi(canny, [(151, 325), (473, 322), (476, 154), (154, 154)])
        
        roi = get_roi(canny, vertices)
        
        rho = cv2.getTrackbarPos('rho', window)
        threshold = cv2.getTrackbarPos('threshold', window)
        min_length = cv2.getTrackbarPos('min length', window)
        max_gap = cv2.getTrackbarPos('max gap', window)
        
        save_value = cv2.getTrackbarPos('save values', window)
        
        if save_value == 1:
            
            houghline_params = {
                
                'rho' : rho, 
                'threshold' : threshold, 
                'min length' : min_length, 
                'max gap' : max_gap
                
                }
            
            print(houghline_params)
        
        # detecting objects 
        
        frame = detect_objects(roi_objects, frame)
        
        lines = cv2.HoughLinesP(roi, 3, np.pi/180, threshold = 25, 
                                minLineLength = 5,
                                maxLineGap = 5)
        
        try:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        except:
            pass 
           
        print(frame)
        cv2.imshow('frame', frame)
        cv2.imshow('objetcs roi',roi_objects)
        cv2.imshow('roi', roi )
        
        if cv2.waitKey(30) == 27:
            break 
        
    cv2.destroyAllWindows()
    cap.release()
    
if __name__ == '__main__':
    main()
    
    