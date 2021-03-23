import cv2
import sys
import os
from cv2 import data
import numpy as np
import pandas as pd
os.environ['DISPLAY'] = ':0'

def show_img(window_name,image):
    print('Correct eyes and faces y/n:')
    cv2.imshow(window_name, image)
    val = input()
    if  val == 'y':
        #cv2.destroyAllWindows()
        return "STORE"

    elif val != 'y':
        #cv2.destroyAllWindows()
        return "NOT STORE"
    '''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return "STORE"
    elif cv2.waitKey(1) & 0xFF == ord('p'):        
        return "NOTSTORE"
    '''

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

def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    coord = []
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
        coord.append((x, y, x + w, y + h))
    return frame, coord

def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    left_eye_coord = []
    right_eye_coord = []
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
            left_eye_coord.append((x, y, x + w, y + h))
        else:
            right_eye = img[y:y + h, x:x + w]
            right_eye_coord.append((x, y, x + w, y + h))
    return left_eye, right_eye, left_eye_coord, right_eye_coord

def frame_faces_extractor(path_file, save_path, class_type):
    vidcap = cv2.VideoCapture(path_file)
    count = 0
    success = 1
    cascPath_face = "Image_creator_script/haarcascade_frontalface_default.xml"
    cascPath_eye = "Image_creator_script/haarcascade_eye.xml"
    faceCascade = cv2.CascadeClassifier(cascPath_face)
    eyeCascade = cv2.CascadeClassifier(cascPath_eye)
    print("Here")
    
    while success:
        
        success,image = vidcap.read()
        if success == False: 
            print("EXTRACTION COMPLETED")
            break
        
        #to detect faces we have to convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        eyes = eyeCascade.detectMultiScale(
            gray
        )

        faces_location = []
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            #crop_img = image[y: y +h, x: x + w]
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            faces_location.append((x , y, x + w, y + h ))
        eye_location = []
        widht = np.size(image)
        for (x_e, y_e, w_e, h_e) in eyes:
            #crop_img = image[y: y +h, x: x + w]
            cv2.rectangle(image, (x_e, y_e), (x_e + w_e, y_e + h_e), (0, 255, 255), 2)
            eye_location.append((x_e , y_e, x_e + w_e, y_e + h_e ))
        
        result = show_img("Video", image)
        print(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if result == "STORE":
            dim = (640, 480)   #DIMESION OF THE RESIZED IMG
            resized = cv2.resize(image, dim)
            
            cv2.imwrite(save_path + "img{}_{}.jpg".format(count ,class_type), resized)     # save frame as JPEG file     
            
            update_csv(save_path + "img{}_{}.jpg".format(count ,class_type), class_type,faces_location, eye_location, "label_final.csv" )
            
            print('Read a n_', count, ' frame: ', success, "path : ", save_path + "img{}_{}.jpg".format(count ,class_type))
            count += 1
        elif result == "NOTSTORE":
            print("Frame Discared")
            continue

    vidcap.release()
    cv2.destroyAllWindows()
    return "STORED"

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

def update_csv(path_file, label, bounding_box, eye_location, save_path):
    df = pd.read_csv(save_path)
    
    temp = {
            'path':path_file,
            'label':label,
            'bounding_box':bounding_box,
            'eye_location': eye_location
        }
    df_temp = pd.DataFrame(data=temp, columns=['path','label','bounding_box','eye_location'])
    if df is not None:
        df_ret = pd.concat(df, df_temp)
        df.to_csv(save_path, index=False)
    else:
        df_temp.to_csv(save_path, index=False)

def create_csv(path, label):
    subdirectories = os.listdir(path)
    ret_vect  = []
    for images in subdirectories:
        temp = {
            'path':path + images,
            'label':label
        }
        ret_vect.append(temp)
    return pd.DataFrame(data=ret_vect, columns=['path','label'])

string_path_badly_mask = 'Image_creator_script/video/badly_masked_{}.mp4'
for i in range(1,4,1):
    frame_faces_extractor(string_path_badly_mask.format(i), "Image_creator_script/img_faces_only/badly_masked_images/{}".format(i), "BADLY_MASKED")


'''
string_path_badly_mask = 'Image_creator_script/video/badly_masked_{}.mp4'
for i in range(1,4,1):
    frame_faces_extractor(string_path_badly_mask.format(i), "Image_creator_script/img_faces_only/badly_masked_images/{}".format(i), "BADLY_MASKED")


#start_capture_two("Image_creator_script/test_video.mp4","Image_creator_script/img_ret/no_mask_img/", "NO-MASK")
print("START")
string_path_mask = 'Image_creator_script/video/mask_{}.mp4'
string_path_no_mask = 'Image_creator_script/video/no_mask_{}.mp4'
for i in range(1,7,1):
    frame_faces_extractor(string_path_mask.format(i),"Image_creator_script/img_faces_only/masked_images/{}".format(i), "MASKED")
    frame_faces_extractor(string_path_no_mask.format(i),"Image_creator_script/img_faces_only/no_mask_images/{}".format(i), "NO-MASKED")

#start_capture_test()
'''

df1 = create_csv('Image_creator_script/img_faces_only/masked_images', "MASKED")
df2 = create_csv('Image_creator_script/img_faces_only/no_mask_images', "NOT_MASKED")
df3 = create_csv('Image_creator_script/img_faces_only/badly_masked_images', "BADLY_MASKED")
df = pd.concat([df1, df2, df3])
df.to_csv('labels.csv', index=False)

