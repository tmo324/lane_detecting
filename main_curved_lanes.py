import cv2
import numpy as np
import math

leftLineBool = None
rightLineBool = None
fps = None
deviation = None

def make_coordinates(image, line_parameters): #return x1, y1, x2, y2
        if np.isnan(np.min(line_parameters)):
                print ("Not detected")
                return
        else:
                slope, intercept = line_parameters

                #print (image.shape)
                y1=image.shape[0]
                y2=int(y1*(3/5))
                x1=int((y1-intercept)/slope)
                x2=int((y2-intercept)/slope)
                if x1<0 or x1>2147483646:
                        x1=0
                if x2<0 or x2>2147483646:
                        x2=0
                return np.array([x1, y1, x2, y2])

def find_curved_points(image, lines):
        line_image = np.zeros_like(image)
        left_points = [] #coord of points
        right_points = []

        left_xy = []
        right_xy = []

        if lines is not None:
                for line in lines:
                        x1, y1, x2, y2 = line.reshape(4)
                        parameters = np.polyfit((x1, x2), (y1, y2), 1)
                        slope=parameters[0]
                        #print("parameters[0]: ", parameters[0])
                        #print("x1, y1, x2, y2: ", x1, y1, x2, y2)
                        intercept = parameters[1]
                        pts = np.array([[x1, y1], [x2 , y2]], np.int32)
                        if slope<-0.4:
                                left_points.append(pts)
                                #left_xy.append([x1, y1])
                                #left_xy.append([x2, y2])
                                
#                                 #yellow
#                                 (b, g, r) = image[y1, x1]
#                                 if r>150 or g>150 or b<100:
#                                     left_xy.append([x1, y1])

                        elif slope> 0.4:
                                right_points.append(pts)
                                #right_xy.append([x1, y1])
                                #right_xy.append([x2, y2])
                                
        #print("left_xy: ", left_xy)
        #print("right_xy: ", right_xy)
        
        
        

#         print("********checking colors now*********")
#         for i in range(len(left_xy)):
#             term = left_xy[i]
#             print(term)
#             (b, g, r) = image[term[1], term[0]]
#             plt.imshow([[(r/255,g/255,b/255)]])
#             plt.show()  
#             print("Color at ", str(term[0])," and ", str(term[1]), " is RGB: ",str(r),str(g),str(b))

# #             im = Image.new('RGB', (500, 300), (128, 128, 128))
# #             draw = ImageDraw.Draw(im)
# #             draw.rectangle((200, 100, 300, 200), fill=(r, g, b), outline=(255, 255, 255))

        
        cv2.polylines(line_image, left_points, True, (0,255,0), 5)
        cv2.polylines(line_image, right_points, True, (0,255,0), 5)

#         #draw the dots
#         x_list = []
#         y_list = []
#         for item in left_xy:        
#                 print(item)
#                 x_list.append(item[0])
#                 y_list.append(item[1])
#                 cv2.drawMarker(line_image, (item[0], item[1]),(0,0,255), markerType=cv2.MARKER_STAR,
#                 markerSize=40, thickness=2, line_type=cv2.LINE_AA)
#         print("x_list:", x_list)
#         print("y_list:", y_list)
#         for item in right_xy:
#                 print(item)
#                 cv2.drawMarker(line_image, (item[0], item[1]),(0,0,255), markerType=cv2.MARKER_STAR,
#                 markerSize=40, thickness=2, line_type=cv2.LINE_AA)

        return line_image


def cannyMaker(image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        canny = cv2.Canny(blur, 30, 150)
        print ("______________Creating canny image___________________")
        return canny

def region_of_interest(image):
        '''
        Mask the given image, given region of interest.
        For now, we focus on a triangle that goes from the bottom to center area of give$
        ''' 
        height = image.shape[0]
        width = image.shape[1]
        print ("______________Masking the image___________________")
        mask=np.zeros_like(image)

            
        #triangle = np.array([
        #[(int(width/9), height),(int(width/2), int(height/10*5.5)),(int(width), height)]
        #])
        #cv2.fillPoly(mask, polygon, 255)
        
        # specify coordinates of the polygon
        polygon = np.array([
            [1, height], 
            [int(width/10*4), int(height/10*5.5)], 
            [int(width/10*6), int(height/10*5.5)], 
            [int(width), height]])
        cv2.fillConvexPoly(mask, polygon, 255)
        
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
    
import cv2
cars_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
def detect_cars(frame):
    cars = cars_cascade.detectMultiScale(frame, 1.15, 6)
    for (x, y, w, h) in cars:
        mid_x = int((x+x+w)/2)
        mid_y = int((y+y+h)/2)
        range_x = int(abs(mid_x-x)*0.75)
        range_y = int(abs(y+h-mid_y)*0.75)
        #print("x, y, w, h: ", x, y, w, h)
        #print("mid_x, mid_y, range_x, range_y: ", mid_x, mid_y, range_x, range_y)
        cv2.rectangle(frame, (mid_x-range_x, mid_y-range_y), (mid_x+range_x,mid_y+range_y), color=(0, 255, 0), thickness=1)
 #       cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=2)

    return frame

def detectLines_and_draw(imageOrVideo):
        '''
        Input: image or video
        Result: Output the video or image with added graphics on it
        '''
        canny_image = cannyMaker(imageOrVideo) #gradient version of the image
        cropped_image=region_of_interest(canny_image) #interested area
        print ("______________Hough Lines Calculated___________________")
        lines = cv2.HoughLinesP(cropped_image,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength=5, maxLineGap=10)
        curved_image = find_curved_points(imageOrVideo, lines)

        final_image= cv2.addWeighted(imageOrVideo, 0.8, curved_image, 1, 1)
        cars_frame = detect_cars(final_image)
        #Show
        cv2.imshow('result', cars_frame)

def main (nameOfTheFile):
        if nameOfTheFile.endswith('.mp4'):
                cap = cv2.VideoCapture(nameOfTheFile)
                if cap.isOpened():
                        print("Device Opened Succesfully\n")
                else:
                        print("Failed to open Device\n")
                while(cap.isOpened()):
                        _, frame = cap.read()
                        global fps
                        fps = round(cap.get(cv2.CAP_PROP_FPS), 2)
                        detectLines_and_draw(frame)
                        if cv2.waitKey(20) == ord('q'):
                                break
                cap.release()
                cv2.destroyAllWindows()
        elif nameOfTheFile.endswith('.jpg'):
                image = cv2.imread(nameOfTheFile)
                lane_image = np.copy(image) #copy of the original image
                detectLines_and_draw(lane_image)
                if cv2.waitKey(0) == ord('q'):
                        cv2.destroyAllWindows()
        else:
                print("other format")

if __name__ == '__main__':
       main("videos/rainy.mp4")
#        main("pics/desert.jpg")

