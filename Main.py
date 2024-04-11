#Note: In a gray scale image 0-->255 means(black-->white)
import cv2
import numpy as np
import utils

webCamFeed = True
pathImage = '1.png'
cap = cv2.VideoCapture(1)
cap.set(10, 640)#ID 10, width of the frame
heightImg = 600
widthImg = 480


utils.initializeTrackbars()
count=0


while True:
    if webCamFeed: 
        #success, img = cap.read()
    #else:
        img = cv2.imread(pathImage)

    img = cv2.resize(img, (widthImg, heightImg)) 
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8) #Create a blank image for testing debuging if required
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convert image to gray scale(common practice)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1) # Add Gaussian BLur(remove noise, smoothen image, EASIER TO FIND CONTOURS)
    thres = utils.valTrackbars() # Get Trackbar values for thresholds
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1]) #Apply Canny blur
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations = 2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations = 1)
    ##Dilation is used to fill in gaps and connect nearby edges that may not been conneted properly by Canny Edge
    ##Erosion is used to refind the edges after dilation

    ##  FIND ALL CONTOURS
    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, hierachy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find all contours(only EXTERNAL contours are found; only endpoints of contours are stored)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10) #DRAW ALL DETECTED CONTOURS

    #Find the biggest contours
    biggest, maxArea = utils.biggestContour(contours) 
    if biggest.size !=0:
        biggest = utils.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0,255,0), 20) #DRAW BIGGEST CONTOUR
        imgBigContour = utils.drawRectangle(imgBigContour, biggest, 2)
        pts1 = np.float32(biggest) #    PREPARE POINTS FOR WARP
        pts2 = np.float32([[0,0],[widthImg, 0], [0, heightImg], [widthImg, heightImg]]) #Prepare points for warp
        #When Using warp perspective, we are telling a point is pts1 corresponds to a point in pts2, the final outcome is pts2(sth like that ;)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        #Remove 20 pixels each size(focus on the content, not the contour)
        imgWarpColored = imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1]- 20]#0 is row, 1 is column
        imgWarpColored = cv2.resize(imgWarpColored, (widthImg, heightImg))

        #APPLY ADAPTIVE THRESHOLD(Used when the lighting condition varies in many parts/regions of the image)
        #Adaptive Thresholding determines the threshold for a PIXEL based on a small region around it, each PIXEL has its OWN threshold value. 
        #Thresholding means seperate elements from its background

        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        #1 means Adaptive_thresh_mean_c(mean of the neighborhood area minus the constant C)
        #1 means thresh Binary(black and white)-Pixels with intensities greater than the threshold are set to maxValue, and all other pixels are set to 0
        #7 is the size of the neigborhood area(7,7)--area size should be odd numbers, 1, 3, 5, 7, 9
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)#Invert binary representation of each pixel(1s-->0s and 0s-->1s)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)#This operation helps to reduce noise and smooth out the image by replacing each pixel's value with the median value in its neighborhood(3 specifies the neighbor size)

        #Image Array for Display
        imageArray = ([img, imgGray, imgThreshold, imgContours],[imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])

    else:
        imageArray = ([img, imgGray, imgThreshold, imgContours],[imgBlank, imgBlank, imgBlank, imgBlank])

    #LABELS FOR DISPLAY
    labels = [["Original","Gray","Threshold","Contours"],["Biggest Contour","Warp Perspective","Warp Gray","Adaptive Threshold"]]

    stackedImage = utils.stackImages(imageArray,0.75,labels)
    cv2.imshow("Result",stackedImage)

    #SAVE IMAGE WHEN "s" is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg", imgWarpColored)
        cv2.rectangle(stackedImage,((int(stackedImage.shape[1]/2)-230), int(stackedImage.shape[0]/2)+50),(1100,350),(0,255,0). cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1]/2)-200,int(stackedImage.shape[0]/2)),cv2.FONT_HERSHEY_DUPLEX, 3, (0,0,255), 5, cv2.LINE_AA)
        cv2.imshow("Result", stackedImage)
        cv2.waitKey(300)
        count += 1