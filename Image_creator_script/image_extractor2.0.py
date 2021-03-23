import cv2
import numpy as np
import pandas as pd
import os
os.environ['DISPLAY'] = ':0'

# init part
face_cascade = cv2.CascadeClassifier("Image_creator_script/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("Image_creator_script/haarcascade_eye.xml")
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)


def detect_faces(img, cascade, face_coord):
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
    #face_coord = []
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
        face_coord.append([x,y,w,h])
    return frame

def detect_eyes(img, cascade, eye_coord):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
            eye_coord.append([x,y,w,h])
        else:
            right_eye = img[y:y + h, x:x + w]
            eye_coord.append([x,y,w,h])
    return left_eye, right_eye

def nothing(x):
    pass

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

def update_csv(path_file, label, bounding_box, eye_location, save_path,file_name, count):
    df_temp = pd.DataFrame(columns=['path','label','bounding_box','eyes_location'])
    df_temp.loc[0, 'path'] = path_file
    df_temp.loc[0, 'label'] = label
    df_temp.loc[0, 'bounding_box'] = bounding_box
    df_temp.loc[0, 'eyes_location'] = eye_location
    print(df_temp.loc[0,:])
    with open(save_path + file_name, 'a') as csv_file:
        df_temp.to_csv(csv_file, header=False, index=False)
    return


def main(path_file, save_path, class_type, i):
    cap = cv2.VideoCapture(path_file)
    count = 0
    success = 1
    while success:
        success, img = cap.read()
        if success == False: 
            print("EXTRACTION COMPLETED")
            break
        dim = (480, 640)   #DIMESION OF THE RESIZED IMG
        frame = cv2.resize(img, dim)
        frame_copy = cv2.resize(img, dim)
        frame_copy = frame.copy()
        face_coord = []
        eyes_coord = []
        discart_frame = False
        face_frame = detect_faces(frame, face_cascade, face_coord)
        
        threshold = 16500 #SET TO A GIVEN AEREA TO DISCARD NOT WANTED VALUES

        for (x, y, w, h) in face_coord:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if (w * h) < threshold:
                discart_frame = True
        if face_frame is not None:
            detect_eyes(frame, eye_cascade, eyes_coord)
            for (x_e,y_e,w_e,h_e) in eyes_coord:
                cv2.rectangle(frame, (x_e, y_e), (x_e + w_e, y_e + h_e), (0, 255, 255), 2)
    
        res_check = len(face_coord) + len(eyes_coord)
        print('shape n', res_check,'detected')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    

        if res_check == 3 and discart_frame == False:
            cv2.imwrite(save_path + "{}img{}_{}.jpg".format(i,count ,class_type), frame_copy)     # save frame as JPEG file     
            if not os.path.exists(save_path + "/temp/"):
                os.makedirs(save_path + "/temp/")
            cv2.imwrite(save_path + "/temp/{}img{}_{}.jpg".format(i,count ,class_type), frame) 
            update_csv(save_path + "{}img{}_{}.jpg".format(i,count ,class_type), 
                    class_type, 
                    face_coord, 
                    eyes_coord, 
                    save_path, 
                    "label_final.csv", 
                    count)
            
            print('Read a n_', count, ' frame: ', success, "path : ", save_path + "{}img{}_{}.jpg".format(i,count ,class_type))
            count += 1
        else:
            print("Frame Discared")
            continue

    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    
    string_path_badly_mask = 'Image_creator_script/video/badly_masked_{}.mp4'
    for i in range(1,4,1):
        main(string_path_badly_mask.format(i), "Image_creator_script/img_faces_only 2.0/badly_masked_images/", "BADLY_MASKED", i)
    string_path_mask = 'Image_creator_script/video/mask_{}.mp4'
    string_path_no_mask = 'Image_creator_script/video/no_mask_{}.mp4'
    for i in range(1,7,1):
        main(string_path_mask.format(i),"Image_creator_script/img_faces_only 2.0/masked_images/", "MASKED", i)
        main(string_path_no_mask.format(i),"Image_creator_script/img_faces_only 2.0/no_mask_images/", "NO-MASKED", i)
    