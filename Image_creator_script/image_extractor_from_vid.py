import cv2
import os
os.environ['DISPLAY'] = ':0'

def show_img(image):
    print('here')
    cv2.imshow('thresh', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def start_capture_two(path_file, save_path, class_type):
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

start_capture_two("Image_creator_script/test_video.mp4","Image_creator_script/img_ret/no_mask_img/", "NO-MASK")
