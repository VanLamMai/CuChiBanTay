import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        # Thêm màu RGB vào nguồn ảnh
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # Xử lý khung ảnh

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    # Vẽ các điểm và nối lại với nhau
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        """Liệt kê vị trí / loại mốc mà chúng tôi đưa
        ra trong danh sách và trong danh sách có lưu
        trữ loại và vị trí của các mốc.
        Liệt kê tất cả các vị trí lmList"""

        lmlist = []

        # Kiểm tra xem các điểm có được phát hiện hay không
        if self.results.multi_hand_landmarks:
            # Kiểm tra xem đang xử lý bàn tay nào
            myHand = self.results.multi_hand_landmarks[handNo]
            # Lấy số id và vẽ đường nối
            for id, lm in enumerate(myHand.landmark):
                # biến id sẽ cung cấp id của điểm ở vị trí chính xác
                # chiều cao, rộng, đường rãnh
                h, w, c = img.shape
                # kiếm vị trí
                cx, cy = int(lm.x * w), int(lm.y * h)  # center
                # print(id,cx,cy)
                lmlist.append([id, cx, cy])

                # Vẽ điểm tròn ở vị trí 0 của đường nối
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmlist


def main():
    # thời gian ban đầu
    pTime = 0
    # thời gian hiện tại
    cTime = 0
    # đọc camera
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        # tính fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # xuất cửa sổ
        cv2.imshow("Video", img)
        # thoát cửa sổ
        if cv2.waitKey(1) == ord('q'):
            print("Thoát camera")
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
