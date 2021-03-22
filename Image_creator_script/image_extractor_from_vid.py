import cv2
import sys
import os
os.environ['DISPLAY'] = ':0'

def show_img(image):
    print('here')
    cv2.imshow('thresh', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extract_frames(path_file, save_path, class_type):
    vidcap = cv2.VideoCapture(path_file)
    count = 0
    success = 1
    while success:
        success,image = vidcap.read()
        if success == False: break
        dim = (640, 480)   #DIMESION OF THE RESIZED IMG
        resized = cv2.resize(image, dim)
        cv2.imwrite(save_path + "img{}_{}.jpg".format(count ,class_type), resized)     # save frame as JPEG file     
        print('Read a n_', count, ' frame: ', success, "path : ", save_path + "img{}_{}.jpg".format(count ,class_type))
        count += 1
    
    vidcap.release()
    cv2.destroyAllWindows()


def frame_faces_extractor(path_file, save_path, class_type):
    vidcap = cv2.VideoCapture(path_file)
    count = 0
    success = 1
    cascPath = "Image_creator_script/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    print("Here")
    while success:
        
        success,image = vidcap.read()
        if success == False: 
            print("EXTRACTION COMPLETED")
            break
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )


        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            crop_img = image[y: y +h, x: x + w]
            #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        dim = (640, 480)   #DIMESION OF THE RESIZED IMG
        resized = cv2.resize(crop_img, dim)
        
        cv2.imwrite(save_path + "img{}_{}.jpg".format(count ,class_type), resized)     # save frame as JPEG file     
        print('Read a n_', count, ' frame: ', success, "path : ", save_path + "img{}_{}.jpg".format(count ,class_type))
        count += 1
    
    vidcap.release()
    cv2.destroyAllWindows()

def start_capture_test():
    cascPath = "Image_creator_script/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

#start_capture_two("Image_creator_script/test_video.mp4","Image_creator_script/img_ret/no_mask_img/", "NO-MASK")
print("START")
frame_faces_extractor("Image_creator_script/test_video_masked.mp4","Image_creator_script/img_faces_only/masked_images/", "MASKED")
frame_faces_extractor("Image_creator_script/test_video_nomask.mp4","Image_creator_script/img_faces_only/no_mask_images/", "NO-MASKED")

#start_capture_test()