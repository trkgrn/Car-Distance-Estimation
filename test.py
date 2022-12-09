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
#img = 'https://i.hizliresim.com/soq91mz.jpg'
img = 'https://images.livemint.com/img/2022/08/04/1600x900/PAKISTAN-ECONOMY-IMPORTS-1_1653543526549_1659575165352_1659575165352.jpg'
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

resultImg = cv2.imread('results/res/abc.jpg', 1)
imgCircle = resultImg.copy()

midPoints = []

for car in boxes:
    x1 = car[0]
    y1 = car[1]
    x2 = car[2]
    y2 = car[3]
    xmid = (x1 + x2) / 2;
    ymid = (y1 + y2) / 2;
    print("X Mid: ", xmid)
    print("Y Mid: ", ymid)
    cv2.circle(imgCircle, center=(int(xmid), int(ymid)), radius=2, color=(0, 0, 255), thickness=10)
    midPoints.append([xmid, ymid])

    # for point in midPoints:
    #     for point2 in midPoints:
    #         if point != point2:
    #             cv2.line(imgCircle, (int(point[0]), int(point[1])), (int(point2[0]), int(point2[1])), color=(0, 255, 0), thickness=4)

startPointX = int(midPoints[0][0])
startPointY = int(midPoints[0][1])
startPoint = (startPointX, startPointY)
for i in range(1, len(midPoints)):
    endPointX = int(midPoints[i][0])
    endPointY = int(midPoints[i][1])
    midX = (startPointX+endPointX) / 2
    midY = (startPointY+endPointY) / 2
    endPoint = (endPointX, endPointY)
    difX = abs(startPointX - endPointX)
    difY = abs(startPointY - endPointY)
    distance = math.sqrt(math.pow(difX, 2) + math.pow(difY, 2))
    print(distance)
    cv2.line(imgCircle, startPoint, endPoint, color=(0, 255, 0), thickness=4)
    cv2.putText(imgCircle, str(int(distance)), (int(midX), int(midY)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('circle', imgCircle)
cv2.waitKey(0)
