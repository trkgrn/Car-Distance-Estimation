import torch
import numpy as np
import yolov5
import cv2
import math
from time import time


class DistanceEstimationDetector:

    def __init__(self, video_path, model_path):
        """
        :param video_path: işlem yapılacak olan video
        :param model_path: işlemleri yapacağımız model
        """
        self.video_path = video_path
        self.model = self.load_model(model_path)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        :return: Seçili video kaydı döndürülür
        """
        return cv2.VideoCapture(self.video_path)

    def load_model(self, model_path):
        """
        Modelin yüklenmesi ve konfigürasyonlarının yapılması
        :param model_path: Yüklenecek modelin dosya yolu
        :return: Yüklenip konfigüre edilmiş model
        """
        model = yolov5.load(model_path)
        model.conf = 0.50  # Tespit edilen objenin güven eşik (confidence threshold) değeri
        model.iou = 0.45  # NMS IoU threshold
        model.agnostic = False  # NMS class-agnostic
        model.multi_label = False  # NMS multiple labels per box
        model.max_det = 1000  # maximum number of detections per image
        model.classes = [2, 5, 7]  # Sadece araçların tespit edilmesi
        return model

    def get_model_results(self, frame):
        """
        Videodan alınan görüntüyü modele sokarak tahmin alıp alınan tahmin sonuçlarını döndürür
        :param frame: videodan alınan görüntü
        :return: modele giren görüntünün sonuçları
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        predictions = results.xyxyn[0]
        cords, scores, labels = predictions[:, :4], predictions[:, 4], predictions[:,5]

        return cords, scores, labels

    def write_rect(self, index, score):
        """
        Rectangle üzerine yazılması gereken obje class name ve threshold değerlerini döndürür
        :param index: Tespit edilen objenin coco.yml 'de karşılık gelen index değeri
        :param score: Tespit edilen objenin confidence threshold değeri
        :return: Rectangle üzerine yazılacak text
        """
        return self.classes[int(index)] + ' ' + str(round(score, 2)) + ' m'

    def draw_rect(self, results, frame):
        """
        Bulunan araçların görüntü üzerinde rectangle içine alınarak belirtilmesi
        :param results: Modelden dönen object detection sonuçları
        :param frame: İşlem yapılan frame
        :return: Tespit edilen objelerin rectangle içine alınmış halinin görüntüsü
        """
        cord, scores, labels = results
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        n = len(labels)  # Tespit edilen obje sayısı

        for i in range(n):
            row = cord[i]
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            green_bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), green_bgr, 1)
        #  cv2.putText(frame, self.write_rect(labels[i], float(scores[i])), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        #              green_bgr, 2)

        return frame

    def calc_distances(self, results, frame):
        """
        Araçlar arasındaki mesafelerin görüntü üzerinden hesaplanması
        :param results: object detection ile edilen sonuç değerleri
        :param frame: işlem yapılacak olan frame
        :return: Mesafe hesaplamalarının yapılıp görüntüye işlenmiş halinin görüntüsü
        """
        cord, scores, labels = results
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        points = []

        for car in cord:
            x1, y1, x2, y2 = int(car[0] * x_shape), int(car[1] * y_shape), int(car[2] * x_shape), int(
                car[3] * y_shape)  # Seçili aracın bulunduğu sınırlar
            x_mid_rect, y_mid_rect = (x1 + x2) / 2, (y1 + y2) / 2  # Eksenlerin orta noktaları
            y_line_length, x_line_length = abs(y1 - y2), abs(x1 - x2)  # Eksenlerin uzunlukları
          #  cv2.circle(frame, center=(int(x_mid_rect), int(y_mid_rect)), radius=1, color=(0, 0, 255), thickness=5)
            points.append([x1, y1, x2, y2, int(x_mid_rect), int(y_mid_rect), int(x_line_length), int(y_line_length)])

        x_shape_mid = int(x_shape / 2)
        start_x, start_y = x_shape_mid, y_shape
        start_point = (start_x, start_y)


        for i in range(0, len(points)):
            end_x1, end_y1, end_x2, end_y2, end_x_mid_rect, end_y_mid_rect, end_x_line_length, end_y_line_length = points[i]
            if end_x2 < x_shape_mid: # Araç solda ise
                end_point = (end_x2, end_y2) # Sağ alt köşeyi seç
            elif end_x1 > x_shape_mid: # Araç sağda ise
                end_point = (end_x1, end_y2) # Sol alt köşeyi seç
            else: # Araç ortada ise
                end_point = (end_x_mid_rect,end_y2) # Alt çizginin ortasını seç

            dif_x, dif_y = abs(start_point[0] - end_point[0]), abs(start_point[1] - end_point[1])
            pixel_count = math.sqrt(math.pow(dif_x, 2) + math.pow(dif_y, 2))
            pixel_per_cm = float(2500 / x_shape)
            distance = float(pixel_count * pixel_per_cm * 1.2 / end_y_line_length)
            #  print(distance)
            cv2.line(frame, start_point, end_point, color=(0, 0, 255), thickness=1)
            cv2.putText(frame, str(round(distance, 2)), (int(end_x1), int(end_y2)), cv2.FONT_HERSHEY_DUPLEX,
                        0.5, (255, 255, 255), 2)

        return frame

    def __call__(self):
        cap = self.get_video_capture()
        assert cap.isOpened()
        while True:
            ret, frame = cap.read()  # Video frame'lere bölünür
            assert ret
            start_time = time()
            results = self.get_model_results(frame)  # Bölünen frame modele girer ve object detection işleminden geçer
            frame = self.draw_rect(results,
                                   frame)  # Elde edilen object detection sonuçları yansıtılacak frame'de çizdirilir
            frame = self.calc_distances(results, frame)  # Tespit edilen araçların arasında mesafelerin bulunması

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)  # FPS hesaplaması
            # print(f"her saniye frame yaz : {fps}")

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('YOLOv5 Distance Estimation', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):  # Sonlandırma için 'Q' tuşuna bas
                break

        cv2.destroyAllWindows()


# DistanceEstimationDetector objesi oluşturulması
detector = DistanceEstimationDetector(video_path='input/car_input.mp4', model_path='yolov5s.pt')
detector()
