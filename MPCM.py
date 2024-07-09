#importing all the required libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cbook
from matplotlib import cm
import matplotlib.colors as colors 
import math
global pcm


#function to read image and determine its height and width
def readImg():
    global pcm
    global mpcmMap
    a = input("Enter the image path: ")
    img = cv2.imread(a,0)
    height,width = img.shape
    newImg = np.asarray(img)
    print("The shape of the image is: ",newImg.shape)
    imgPadding(newImg,height,width)
    #defing a global 3d array that will store pcm for each scale and mpcm for the entire frame
    mpcmMap = np.zeros((height,width))
    pcm = np.zeros((4,height,width))
    print(pcm.shape)

#function to perform Mean filtering (smoothig) in our image    
def meanFiltering(new_matrix):
    #selecting a 9*9 kernel for mean filtering
    ksize = (9,9)
    new_matrix = cv2.blur(new_matrix, ksize)
        
#function to turm image into square matrix
def imgPadding(newImg, height, width):
    size = max(height, width)
    new_matrix = np.zeros((size, size))
    new_matrix[:height, :width] = newImg
    #new_matrix is now a square matrix
    meanFiltering(new_matrix)
    finalImgPadding(new_matrix)
    
#function to do padding for different scales and window sizes 3x3, 5x5, 7x7 and 9x9
def finalImgPadding(new_matrix):
    matrix3 = new_matrix
    np.pad(matrix3, pad_width= 4, mode='constant',constant_values=0)
    PCM(matrix3,3,0)

#function to calculate pcm for the image at a certain scale
def PCM(img, windowSize,scale):
    global pcm
    row,column = img.shape
    window = np.zeros((windowSize*windowSize))
    i,j,m,n,l,t=0,0,0,0,0,0
    #loops to move the sliding window with a stride of 1px and each subframe is of 9X9
    while i < row-(9*windowSize):
        while j < column-(9*windowSize):
            #patch = sliding window at he moment
            patch = img[j:(j+(9*windowSize)),i:(i+(9*windowSize))]
            temp = np.zeros(windowSize*windowSize)
            d = np.zeros(windowSize*windowSize)
            D = np.zeros(math.floor((windowSize*windowSize)/2))
            while m < i+(9*windowSize):
                while n < j+(9*windowSize):
                    subpatch = patch[n:n+9, m:m+9]
                    #storing the value of max gray value of our target subframe in l
                    if m == math.ceil(windowSize/2) and n == math.ceil(windowSize/2):
                        l=max(subpatch.flatten())
                    #calculating mean gray value of each subframe and storing it in an array temp
                    subpatchAvg = subpatch.flatten()
                    subpatchAvg = np.sum(subpatchAvg)/len(subpatchAvg)
                    print(m,n)
                    temp[t] = subpatchAvg
                    t+=1
                    n+=9
                m+=9
            for i in range(0,(windowSize*windowSize)):
                window[i] = l/temp[i]
                d[i] = temp[math.floor(len(temp)/2)] - temp[i]
            for  a in range(0,math.floor((windowSize*windowSize)/2)):
                if(d[a]>0 and d[len(d)-a-1]>0):
                    D[a] = d[a]*d[len(d)-a-1]
                else:
                    D[a] = 0
            PatchBasedConstrastMeasure = min(D)
            pcm[scale][i][j] = PatchBasedConstrastMeasure
            j+=1
        i+=1
    print("The PCM of the scale = ",scale," looks like:-")
    plotGraph(pcm[0])
    
#function to generate mpcm map of a given image
def mpcmGenerator():
    p,q = mpcmMap.shape
    for t1 in range(0,p):
        for t2 in range(0,q):
            mpcmMap[t1][t2] = max(pcm[0][p][q],pcm[1][p][q],pcm[2][p][q],pcm[3][p][q])

#fucntion for target detection using mpcm
def targetDetection():
    p,q = mpcmMap.shape
    arr1 = mpcmMap.flatten()
    avgMPCM = np.sum(arr1)/(p*q)
    #calculating std deviation
    std_deviation = 0
    for i in range(0,p):
        for j in range(0,q):
            std_deviation += (mpcmMap[i][j]-avgMPCM)
    std_deviation = std_deviation/(p*q)
    std_deviation = math.sqrt(std_deviation)
    #taking the value of k in adaptive thresholding as 10
    k = 10
    threshold = avgMPCM + (k*std_deviation)
    print("MPCM map before thresholding")
    plotGraph(mpcmMap)
    #Applying thresholding in the MPCM map
    for i in range(0,p):
        for j in range(0,q):
            if mpcmMap[i][j] >= threshold:
                mpcmMap[i][j] = 1
            else:
                mpcmMap[i][j] = 0
    print("MPCM map after thresholding")
    plotGraph(mpcmMap)

#fucntion to print matrices as 3d bar graphs
def plotGraph(matrix):
    # Generate X and Y coordinates for the matrix values
    x = np.arange(matrix.shape[1])
    y = np.arange(matrix.shape[0])
    x, y = np.meshgrid(x, y)
    # Create a figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the surface
    surf = ax.plot_surface(x, y, matrix, cmap='viridis')
    # Customize the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('gray level')
    plt.colorbar(surf)
    # Show the plot
    plt.show()
    
    
#Driver code for the MPCM program
print("welcom to the mpcm program")
readImg()

    