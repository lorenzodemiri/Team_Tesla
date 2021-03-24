# haar cascade to extract faces from video
import cv2
class FaceDetector:
    def __init__(self, bottom, top):
        self.bottom = bottom
        self.top = top
    def detect(self):
        frontalface_default = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        frontalface_alt2 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        profileface = cv2.CascadeClassifier('haarcascade_profileface.xml')
        cap = cv2.VideoCapture(0)
        f = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True and f%10==0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frontalfaces_default = frontalface_default.detectMultiScale(gray_frame, 1.03, 1)
                frontalfaces_alt2 = frontalface_alt2.detectMultiScale(gray_frame,1.03,1)
                profilefaces = profileface.detectMultiScale(gray_frame,1.03,1)
                for (x,y,w,h) in frontalfaces_default:
                    if w & h > self.bottom and w & h < self.top:
                        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0))
                        face = frame[y:y+h,x:x+w]
                        face_resized = cv2.resize(face, (224,224), interpolation=cv2.INTER_AREA)
                        # predict here on face_resized
                for (x,y,w,h) in frontalfaces_alt2:
                    if w & h > self.bottom and w & h < self.top:
                        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0))
                        face = frame[y:y+h,x:x+w]
                        face_resized = cv2.resize(face, (224,224), interpolation=cv2.INTER_AREA)
                        # predict here on face_resized
                for (x,y,w,h) in profilefaces:
                    if w & h > self.bottom and w & h < self.top:
                        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0))
                        face = frame[y:y+h,x:x+w]
                        face_resized = cv2.resize(face, (224,224), interpolation=cv2.INTER_AREA)
                        # predict here on face_resized
                cv2.imshow('face', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            f+=1
        cap.release()
        cv2.destroyAllWindows()
