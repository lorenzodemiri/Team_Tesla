import cv2
import sys
import os
import pandas as pd
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
        
        #to detect faces we have to convert to grayscale
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

