import cv2
import numpy as np
import math

leftLineBool = None
rightLineBool = None
fps = None
deviation = None

def region_of_interest_to_see(image, lines): #draws the area that the car can travel forward
	height = image.shape[0]
	width = image.shape[1]

	line_image = np.zeros_like(image)
	pts = np.array([
	[(int(width/10), height),(int(width/2), int(height/10*4)),(int(width/10*9), height)]
	])
	cv2.fillPoly(line_image, pts, (0, 255, 0))
	return line_image

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

#https://answers.opencv.org/question/185375/how-do-i-detect-the-curvy-lines-in-opencv/

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
			intercept = parameters[1]
			pts = np.array([[x1, y1], [x2 , y2]], np.int32)
			if slope<0:
				left_points.append(pts)
				left_xy.append([x1, y1])
				left_xy.append([x2, y2])
			else:
				right_points.append(pts)
				right_xy.append([x1, y1])
				right_xy.append([x2, y2])

	print("left_points: ", left_points)
	print("right_points: ", right_points)
	print("left_xy: ", left_xy)
	print("right_xy: ", right_xy)	

	cv2.polylines(line_image, left_points, True, (0,255,0))
	cv2.polylines(line_image, right_points, True, (0,255,0))


	# for item in left_xy:
	# 	print(item)
	# 	cv2.drawMarker(line_image, (item[0], item[1]),(0,0,255), markerType=cv2.MARKER_STAR, 
	# 	markerSize=40, thickness=2, line_type=cv2.LINE_AA)






	return line_image


def cannyMaker(image):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	canny = cv2.Canny(blur, 50, 150)
	return canny

def region_of_interest(image):
	'''
	Mask the given image, given region of interest.
	For now, we focus on a triangle that goes from the bottom to center area of given image or video
	'''
	height = image.shape[0]
	width = image.shape[1]
	print ("_________________________________")
	print (height)
	print (width)
	
	triangle = np.array([
	[(int(width/9), height),(int(width/2), int(height/10*5.5)),(int(width/10*9), height)]
	])

	mask=np.zeros_like(image)
	cv2.fillPoly(mask, triangle, 255)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def detectLines_and_draw(imageOrVideo):
	'''
	Input: image or video 
	Result: Output the video or image with added graphics on it
	'''

	canny_image = cannyMaker(imageOrVideo) #gradient version of the image
	cropped_image=region_of_interest(canny_image) #interested area
	#lines calculations
	#lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
	lines = cv2.HoughLinesP(cropped_image,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength=40, maxLineGap=5)
	#averaged_lines = find_average_slope_intercept(imageOrVideo, lines) #slope and intercept average of the lines, or plotted dots.
	curved_image = find_curved_points(imageOrVideo, lines)

	combo_image= cv2.addWeighted(imageOrVideo, 0.8, curved_image, 1, 1)
	#Show
	cv2.imshow('result', combo_image)




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
	main("videos/seattlestreet2.mp4")


