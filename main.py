try:
    import os
    import traceback
    import numpy as np
    import cv2
    import mediapipe as mp
    from os.path import dirname, abspath, join
except ImportError as ie:
    print(ie)

# project dir
project_dir = abspath(dirname(__name__))
media_dir = join(project_dir, 'media')
input_file = join(media_dir, '01.mp4')

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

prev_zone = None
curr_zone = None
display = None
window_name = "tracking window"

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

    writer = cv2.VideoWriter(filename='tracked video.mp4',
                             fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                             fps=25,
                             frameSize=(width, height))
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while capt.isOpened():
            ret, frame = capt.read()

            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                nose_x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * width
                nose_y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * height

                if (nose_x > red_start) & (nose_x < red_end):
                    if curr_zone == None:
                        curr_zone = 'red'
                    else:
                        if curr_zone == 'green':
                            prev_zone = curr_zone
                            curr_zone = 'red'

                            display = "Good Bye!"

                elif (nose_x > green_start) & (nose_x < green_end):
                    if curr_zone == None:
                        curr_zone = 'green'
                    else:
                        if curr_zone == 'red':
                            prev_zone = curr_zone
                            curr_zone = 'green'

                            display = 'Welcome!'

                else:
                    prev_zone = curr_zone
                    curr_zone = None
                    display = None
            except:
                pass

            # Drawing pose landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=1)
                                      )

            if display:
                cv2.putText(frame, text=display, org=(10, 50), fontFace=font, fontScale=2, color=(0,0,0),
                            thickness=2, lineType=cv2.LINE_AA)

            # Red zone
            frame[:, red_start:red_end, :] = cv2.addWeighted(src1=frame[:, red_start:red_end, :],
                                                           alpha=0.8, src2=red_arr, beta=0.2, gamma=0)

            # Green Zone
            frame[:, green_start:green_end, :] = cv2.addWeighted(src1=frame[:, green_start:green_end, :],
                                                           alpha=0.8, src2=green_arr, beta=0.2, gamma=0)

            writer.write(frame)
            cv2.imshow(window_name, frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
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