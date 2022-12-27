import cv2

if __name__ == '__main__':
    # img = cv2.imread("/home/duclam/Downloads/lua.jpeg", cv2.IMREAD_COLOR)
    img = cv2.imread("/home/duclam/Documents/dataset_action/UR_Fall_dataset/Fall/Cam_1/fall-28-cam1-rgb/fall-28-cam1-rgb-003.png", cv2.IMREAD_COLOR)
    # x1 = 2
    # y1 = 22
    # x2 = 235
    # y2 = 352
    x1, y1, x2, y2 = 366 ,263 ,541, 431
    # y1 = 253
    # x2 = 386
    # y2 = 452
    frame = cv2.rectangle(img, (x1,y1)  , (x2,y2) , (255,0,0) , 2)
    scale_percent = 100 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
   
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    print('dim :', dim)
    
    # resize image
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("image", frame)
    cv2.waitKey(0)
    if 0xFF == ord('q') :
        exit
 
# It is for removing/deleting created GUI window from screen
# and memory
    cv2.destroyAllWindows()