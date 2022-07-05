try:
    import os
    import math
    import traceback
    import numpy as np
    import cv2
    import torch
    from os.path import dirname, abspath, join
except ImportError as ie:
    print(ie)

# project dir
project_dir = abspath(dirname(__name__))
media_dir = join(project_dir, 'media')
input_file = join(media_dir, '03.mp4')

prev_zone = None
curr_zone = None
display = None
window_name = "tracking window"

cp_prev_frame = []
tracking_object = {}
tracking_id = 0
first_run = True

try:
    capt = cv2.VideoCapture(input_file)

    width = int(capt.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capt.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)

    red_start = round((width / 100) * 20)
    red_end = round((width / 100) * 50)
    green_start = round((width / 100) * 50)
    green_end = round((width / 100) * 80)

    empty_arr = np.zeros(shape=(height, red_end - red_start, 3), dtype=np.uint8)
    red_arr = empty_arr.copy()
    green_arr = empty_arr.copy()

    red_arr[:, :, 2] = 255
    green_arr[:, :, 1] = 255

    font = cv2.FONT_HERSHEY_SIMPLEX

    # saving video
    writer = cv2.VideoWriter(filename='tracked video1.mp4',
                             fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                             fps=25,
                             frameSize=(width, height))
    # full screen
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Custom yolo model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./yolov5/runs/train/exp/weights/best.pt')

    # model params
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold

    while capt.isOpened():
        ret, frame = capt.read()

        # break out of while loop if video ends
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        roi = frame_rgb[:, red_start:green_end, :].copy()

        results = model(roi)
        predictions = results.pred[0]
        boxes = predictions[:, :4] #x1, y1, x2, y2

        cp_cur_frame = []

        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            cx = int((x1+x2) / 2)
            cy = int((y1+y2) / 2)

            cp_cur_frame.append((cx, cy))

            #cv2.circle(roi, (cx,cy), 3, (0,0,255), -1)
            cv2.rectangle(roi, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)


        if first_run:
            for cx1, cy1 in cp_cur_frame:
                for cx2, cy2 in cp_prev_frame:
                    distance = math.hypot(cx2-cx1, cy2-cy1)

                    if distance < 25:
                        tracking_object[tracking_id] = (cx1,cy1)
                        tracking_id += 1

            if len(tracking_object) != 0:
                first_run = False
        else:
            tracking_object_copy = tracking_object.copy()
            cp_cur_frame_copy = cp_cur_frame.copy()

            for obj_id, pt2 in tracking_object_copy.items():
                object_exists = False
                for pt in cp_cur_frame_copy:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                    # update object position
                    if distance < 25:
                        tracking_object[obj_id] = pt
                        object_exists = True

                        if pt in cp_cur_frame:
                            cp_cur_frame.remove(pt)
                        continue

                # remove id
                if not object_exists:
                    tracking_object.pop(obj_id)

            for pt in cp_cur_frame:
                tracking_object[tracking_id] = pt
                tracking_id += 1

        for obj_id, pt in tracking_object.items():
            cv2.circle(roi, pt, 3, (0,0,255), -1)
            cv2.putText(roi, str(obj_id), (pt[0], pt[1] - 7), 0, 0.5, (0,0,255),1)

        if display:
            cv2.putText(frame, text=display, org=(10, 50), fontFace=font, fontScale=2, color=(0,0,0),
                        thickness=2, lineType=cv2.LINE_AA)

        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        frame[:, red_start:green_end, :] = roi
        # Red zone
        frame[:, red_start:red_end, :] = cv2.addWeighted(src1=frame[:, red_start:red_end, :],
                                                       alpha=0.8, src2=red_arr, beta=0.2, gamma=0)

        # Green Zone
        frame[:, green_start:green_end, :] = cv2.addWeighted(src1=frame[:, green_start:green_end, :],
                                                       alpha=0.8, src2=green_arr, beta=0.2, gamma=0)

        # cp previous frame
        cp_prev_frame = cp_cur_frame.copy()

        writer.write(frame)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except (Exception, KeyboardInterrupt) as ex:
    print(repr(ex))
    tb = traceback.extract_tb(ex.__traceback__)
    for t in tb:
        t = list(t)
        if not any([substr in str(t[0]) for substr in ['_libs', 'site-packages']]):
            print(f"Error details: file {t[0]} at line {t[1]} in {t[2]}")

finally:
    capt.release()
    writer.release()
    cv2.destroyAllWindows()