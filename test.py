import yolov5
import cv2
import math

# load pretrained model
model = yolov5.load('yolov5s.pt')

# set model parameters
model.conf = 0.50  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image
model.classes = [2]

# set image
#img = 'input/input.png'
img = 'input/input.png'
# perform inference
results = model(img)

# inference with larger input size
results = model(img, size=1280)

# inference with test time augmentation
results = model(img, augment=True)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4]  # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# print(predictions[:,0]) # x1 değerleri
# print(predictions[:,1]) # y1 değerleri
# print(predictions[:,2]) # x2 değerleri
# print(predictions[:,3]) # y2 değerleri
# print(predictions[:,4]) # skorlar
# print(predictions[:,5]) # kategoriler


# show detection bounding boxes on image
results.show()

# save results into "results/" folder
results.save(save_dir='results/res')

resultImg = cv2.imread('results/res/input.jpg', 1)
imgCircle = resultImg.copy()
x_shape, y_shape = resultImg.shape[1], resultImg.shape[0]
print(x_shape,y_shape)
midPoints = []



for car in boxes:
    x1 = car[0]
    y1 = car[1]
    x2 = car[2]
    y2 = car[3]
    xmid = (x1 + x2) / 2;
    ymid = (y1 + y2) / 2;
    # print("X Mid: ", xmid)
    # print("Y Mid: ", ymid)
    # cv2.circle(imgCircle, center=(int(xmid), int(ymid)), radius=2, color=(0, 0, 255), thickness=10)
    midPoints.append([xmid, ymid,x2,y2,x1,y1])

    # for point in midPoints:
    #     for point2 in midPoints:
    #         if point != point2:
    #             cv2.line(imgCircle, (int(point[0]), int(point[1])), (int(point2[0]), int(point2[1])), color=(0, 255, 0), thickness=4)

x_shape_mid = int(x_shape / 2)
startPointX, startPointY = x_shape_mid, y_shape
startPoint = (startPointX, startPointY)
width_in_rf = 121
measured_distance = 275  #inch =700cm
real_width = 60 #inch = 150 cm
focal_length = (width_in_rf * measured_distance) / real_width

for i in range(0, 2):
    endPointX = int(midPoints[i][0])
    endPointY = int(midPoints[i][1])
    endPointX2 = int(midPoints[i][2])
    endPointY2 = int(midPoints[i][3])
    endPointX1 = int(midPoints[i][4])
    endPointY1 = int(midPoints[i][5])

    midX = (startPointX+endPointX) / 2
    midY = (startPointY+endPointY) / 2
    endPoint = (endPointX2, endPointY2)
    difX = abs(startPointX - endPointX2)
    difY = abs(startPointY - endPointY2)
    pixel_count = math.sqrt(math.pow(difX, 2) + math.pow(difY, 2))
    print(abs(endPointX1-endPointX2))
    distance = real_width * focal_length / abs(endPointY1-endPointY2);
    distance = distance * 2.54
    cv2.line(imgCircle, startPoint, endPoint, color=(0, 255, 0), thickness=2)
    cv2.putText(imgCircle, str(int(distance)), (int(midX), int(midY)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('circle', imgCircle)
cv2.waitKey(0)
