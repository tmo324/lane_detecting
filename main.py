import cv2
import numpy as np
import math

leftLineBool = None
rightLineBool = None
fps = None
deviation = None

def display_lines(image, lines): #draws lines
	line_image = np.zeros_like(image)
	print (lines, "lines")
	global leftLineBool
	global rightLineBool

	if lines[0, 2] == 0 and lines[1,2] == 0:
		print ("Both list empty DL")
		leftLineBool=False
		rightLineBool=False
	elif lines[0, 2] == 0:
		print ("the LEFT list is empty DL")
		cv2.line(line_image, (lines[1,0], lines[1,1]), (lines[1,2], lines[1,3]), (255, 255, 0), 10)
		leftLineBool=False
		rightLineBool=True
	elif lines[1,2] == 0:
		print ("the RIGHT list is empty DL")
		cv2.line(line_image, (lines[0,0], lines[0,1]), (lines[0,2], lines[0,3]), (255, 255, 0), 10)
		leftLineBool = True
		rightLineBool=False
	elif lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 0), 10)
			leftLineBool = True
			rightLineBool = True
	else:
		print ()
	return line_image

def display_polygon(image, lines): #draws the area that the car can travel forward
	line_image = np.zeros_like(image)

	if lines[0, 2] == 0 and lines[1,2] == 0:
		print("Both empty DP. Will not draw the green area!")
	elif lines[0, 2] == 0:
		print ("the LEFT list is empty DP. Cant determine the green area!")
	elif lines[1,2] == 0:
		print ("the RIGHT list is empty DP. Cant determine the green area!")
	else:
		pts = np.array([
		[(lines[0,0], lines[0,1]),(lines[0,2], lines[0,3]), (lines[1,2], lines[1,3]),(lines[1,0], lines[1,1])]
		])
		cv2.fillPoly(line_image, pts, (0, 255, 0))
	return line_image

def display_the_center(image, lines): #draws the area that the car can travel forward
	height = image.shape[0]
	line_image = np.zeros_like(image)
	font = cv2.FONT_HERSHEY_SIMPLEX

	if lines[0, 2] == 0 and lines[1,2] == 0:
		print("Both empty FTC. Will not draw the center!")
	elif lines[0, 2] == 0:
		print ("the LEFT list is empty FTC. Cant determine the center")
	elif lines[1,2] == 0:
		print ("the RIGHT list is empty FTC, Cant determine the center")
	else:
		averageX0=int((lines[0,2]+lines[1,2])/2)
		averageY0=int((lines[0,3]+lines[1,3])/2)
		averageX1=int((lines[0,0]+lines[1,0])/2)
		averageY1=int((lines[0,1]+lines[1,1])/2)
		cv2.line(line_image, (averageX1, height), (averageX1, height-50), (255, 255, 255), 5)
		cv2.putText(line_image, "Center", (averageX1, height-50), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
	return line_image

def display_message(image, leftLineStatus, rightLineStatus, fps, dev):
	line_image = np.zeros_like(image)
	cv2.rectangle(line_image, (20, 20), (200, 100), (255, 255, 255), -1)
	font = cv2.FONT_HERSHEY_SIMPLEX
	fps_message = "FPS: " + str(fps)
	dev_message = "Deviation: " + str(dev)

	if rightLineStatus == False and leftLineStatus == False:
		cv2.putText(line_image, "Left: Not Found", (20, 35), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
		cv2.putText(line_image, "Right: Not found", (20, 55), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
	elif leftLineStatus == False:
		cv2.putText(line_image, "Left: Not Found", (20, 35), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
		cv2.putText(line_image, "Right: Detected", (20, 55), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
	elif rightLineStatus == False:
		cv2.putText(line_image, "Left: Detected", (20, 35), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
		cv2.putText(line_image, "Right: Not found", (20, 55), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
	else:
		cv2.putText(line_image, "Left: Detected", (20, 35), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
		cv2.putText(line_image, "Right: Detected", (20, 55), font, 0.5, (0,255,0), 1, cv2.LINE_AA)

	cv2.putText(line_image, fps_message, (20, 75), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
	cv2.putText(line_image, dev_message, (20, 95), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
	return line_image

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

def find_average_slope_intercept(image, lines):
	left_fit = [] #coord of points
	right_fit = []
	print (lines)

	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			parameters = np.polyfit((x1, x2), (y1, y2), 1)
			slope=parameters[0]
			intercept = parameters[1]
			if slope<0:
				left_fit.append((slope, intercept))
			else:
				right_fit.append((slope, intercept))
	left_fit_average = np.average(left_fit, axis=0)
	right_fit_average = np.average(right_fit, axis=0)
	
	left_line = make_coordinates(image, left_fit_average)
	if left_line is None:
		print ("Unable to detect LEFT line")
		left_line=[0,0,0,0]
	print (left_line, "left line")

	right_line = make_coordinates(image, right_fit_average)
	if right_line is None:
		print ("Unable to detect RIGHT line")
		right_line=[0,0,0,0]
	print (right_line, "right line")

	return np.array([left_line, right_line])

def cannyMaker(image):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	canny = cv2.Canny(blur, 50, 150)
	return canny

def region_of_interest(image):
	height = image.shape[0]
	width = image.shape[1]
	print ("_________________________________")
	print (height)
	print (width)
	
	triangle = np.array([
	[(int(width/9), height),(int(width/2), int(height/10*5)),(int(width/10*9), height)]
	])

	mask=np.zeros_like(image)
	cv2.fillPoly(mask, triangle, 255)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def detectLines_and_draw(imageOrVideo):
	canny_image = cannyMaker(imageOrVideo) #gradient version of the image
	cropped_image=region_of_interest(canny_image) #interested area
	#lines calculations
	lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
	averaged_lines = find_average_slope_intercept(imageOrVideo, lines) #slope and intercept average of the lines, or plotted dots.
	#display definitions
	line_image=display_lines(imageOrVideo, averaged_lines) #draw the lines
	polygon_image=display_polygon(imageOrVideo, averaged_lines) #draw the green safe area
	center_line=display_the_center(imageOrVideo, averaged_lines)
	message = display_message(imageOrVideo, leftLineBool, rightLineBool, fps, deviation)
	#Draw the lines and the detected green area on the original pics
	combo_image= cv2.addWeighted(imageOrVideo, 0.8, polygon_image, 1, 1)
	combo1_image= cv2.addWeighted(combo_image, 0.8, line_image, 1, 1)
	combo2_image= cv2.addWeighted(combo1_image, 0.8, center_line, 1, 1)
	combo3_image= cv2.addWeighted(combo2_image, 0.8, message, 1, 1)
	#Show
	cv2.imshow('result', combo3_image)



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



main("videos/desert.mp4")