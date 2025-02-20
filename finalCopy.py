import copy
import csv
import itertools
from collections import Counter, deque
import ctypes

import cv2 as cv
import mediapipe as mp
import numpy as np
import pydirectinput
from time import sleep

from model import KeyPointClassifier, PointHistoryClassifier
from KeyboardController import Mouse, Keyboard


def main():
    WIDTH = ctypes.windll.user32.GetSystemMetrics(1)
    HEIGHT = ctypes.windll.user32.GetSystemMetrics(0)

    cap_width = 960
    cap_height = 540

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.95,
    )

    keypoint_classifier = KeyPointClassifier()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv',encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    prev_wrists = None
    mouse = Mouse()
    keyboard = Keyboard()

    mouse.set(WIDTH//2, HEIGHT//2)

    click_left = False
    click_right = False
    key_w = False
    key_s = False
    key_d = False
    key_a = False

    while True:
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                if handedness.classification[0].label == "Right":
                    wrist = landmark_list[0]
                    if (prev_wrists != None) and (hand_sign_id == 1):
                        speed = (wrist[0] - prev_wrists[0], wrist[1] - prev_wrists[1])
                        sensetivity = 3
                        pydirectinput.move(int((speed[0])*sensetivity/WIDTH*cap_width), int((speed[1])*sensetivity/HEIGHT*cap_height)) #optimal settings for mine craft
                        #pydirectinput.move(int((speed[0])), int((speed[1]))) #optimal settings for roblox

                    prev_wrists = wrist
                if handedness.classification[0].label == "Right" and hand_sign_id == 2:
                    direction = calc_direction(landmark_list[8], landmark_list[5], hand_sign_id)
                    if direction == "left":
                        pydirectinput.leftClick()
                    if direction == "right":
                        pydirectinput.rightClick()
                if handedness.classification[0].label == "Left" and hand_sign_id == 2:
                    direction = calc_direction(landmark_list[8], landmark_list[5], hand_sign_id)
                    if direction == "bottom":
                        if key_s == False:
                            #keyboard.hold(keyboard.Key.s)
                            pydirectinput.keyDown('s')
                            key_s = True
                    else:
                        key_s = False
                        #keyboard.release(keyboard.Key.s)
                        pydirectinput.keyUp('s')

                    if direction == "top":
                        if key_w == False:
                            #keyboard.hold(keyboard.Key.w)
                            pydirectinput.keyDown('w')
                        key_w = True
                    else:
                        key_w = False
                        #keyboard.release(keyboard.Key.w)
                        pydirectinput.keyUp('w')

                    if direction == "left":
                        if key_a == False:
                            #keyboard.hold(keyboard.Key.a)
                            pydirectinput.keyDown('a')
                        key_a = True
                    else:
                        key_a = False
                        #keyboard.release(keyboard.Key.a)
                        pydirectinput.keyUp('a')

                        
                    if direction == "right":
                        if key_d == False:
                            #keyboard.hold(keyboard.Key.d)
                            pydirectinput.keyDown('d')
                        key_d = True
                    else:
                        key_d = False
                        #keyboard.release(keyboard.Key.d)
                        pydirectinput.keyUp('d')


                if handedness.classification[0].label == "Left" and hand_sign_id == 0:
                    pydirectinput.press('space')
                    key_d = False
                    key_w = False
                    key_a = False
                    key_s = False   
                    keyboard.release(keyboard.Key.d)
                    keyboard.release(keyboard.Key.a)
                    keyboard.release(keyboard.Key.s)
                    keyboard.release(keyboard.Key.w) 
                     
                if handedness.classification[0].label == "Left" and hand_sign_id == 3:
                    pydirectinput.press('e') 
                    key_d = False
                    key_w = False
                    key_a = False
                    key_s = False   
                    keyboard.release(keyboard.Key.d)
                    keyboard.release(keyboard.Key.a)
                    keyboard.release(keyboard.Key.s)
                    keyboard.release(keyboard.Key.w)
                if handedness.classification[0].label == "Left" and hand_sign_id == 1:
                    key_d = False
                    key_w = False
                    key_a = False
                    key_s = False   
                    pydirectinput.keyUp('w')
                    pydirectinput.keyUp('a')
                    pydirectinput.keyUp('d')
                    pydirectinput.keyUp('s')

                
                debug_image = draw_bounding_rect(debug_image, brect)
                debug_image = draw_landmark(debug_image, results.multi_hand_landmarks, mp_drawing, mp_hands, mp_drawing_styles)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                )
        cv.imshow('Hand Gesture Recognition', debug_image)

def calc_direction(tip_index, root_index, hand_sign_id):
    x_dir = root_index[0] - tip_index[0]
    y_dir = root_index[1]-tip_index[1]

    unit_vect = (1,0)
    try:
        distance_vect = (x_dir/x_dir, y_dir/y_dir)

        dot = np.dot(distance_vect, unit_vect)
        rad = np.arccos(dot)
        deg = np.degrees(rad)
        deg = np.degrees(np.arctan2(y_dir, x_dir))

        if (-180<=deg<=-150) or (150<=deg<=180):
            return "right"
        if (-30<=deg<=30):
            return "left"
        if (-120<=deg<=-60):
            return "bottom"
        if (60<=deg<=120):
            return "top"
    except:
        if x_dir == 0:
            if y_dir >= 0:
                return "top"
            else:
                return "bottom"
        else:
            if x_dir >= 0:
                return "right"
            else:
                return "left"
    return None

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def draw_landmark(image,landmarks, mp_drawing, mp_hands, mp_drawing_styles):
    for hand_landmarks in landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    return image

def draw_bounding_rect(image, brect):
    # Outer rectangle
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                 (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0: ]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image


if __name__ == "__main__":
    main()