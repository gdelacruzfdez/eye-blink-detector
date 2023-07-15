import cv2
import dlib
import math
import numpy as np

# Constants
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
BOX_COLOR = (0, 255, 0)  # Green color for the bounding boxes
BOX_THICKNESS = 2

# Initialize dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

def calculate_distance(landmark1, landmark2):
    x1, y1 = landmark1
    x2, y2 = landmark2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(landmark1, landmark2):
    x1, y1 = landmark1
    x2, y2 = landmark2
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def extract_rotated_box(img, box, angle):
    if len(box) == 0:
        return None

    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    img_crop = img_rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]

    return img_crop

def get_eye_landmarks(landmarks, start, end):
    eye_landmark1 = (landmarks.part(start).x, landmarks.part(start).y)
    eye_landmark2 = (landmarks.part(end).x, landmarks.part(end).y)
    return eye_landmark1, eye_landmark2

def get_eye_box(eye_landmark1, eye_landmark2, eye_distance):
    eye_center = ((eye_landmark1[0] + eye_landmark2[0]) // 2, (eye_landmark1[1] + eye_landmark2[1]) // 2)
    eye_width = 2 * eye_distance
    eye_height = eye_distance
    eye_angle = calculate_angle(eye_landmark1, eye_landmark2)
    eye_box = cv2.boxPoints(((eye_center[0], eye_center[1]), (eye_width, eye_height), eye_angle))
    return np.int0(eye_box), eye_angle

def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_landmark1, left_eye_landmark2 = get_eye_landmarks(landmarks, 36, 39)
        right_eye_landmark1, right_eye_landmark2 = get_eye_landmarks(landmarks, 42, 45)

        left_eye_distance = calculate_distance(left_eye_landmark1, left_eye_landmark2)
        right_eye_distance = calculate_distance(right_eye_landmark1, right_eye_landmark2)

        left_eye_box, left_eye_angle = get_eye_box(left_eye_landmark1, left_eye_landmark2, left_eye_distance)
        right_eye_box, right_eye_angle = get_eye_box(right_eye_landmark1, right_eye_landmark2, right_eye_distance)

        left_eye_image = extract_rotated_box(frame, left_eye_box, left_eye_angle)
        right_eye_image = extract_rotated_box(frame, right_eye_box, right_eye_angle)

        cv2.imshow('Left Eye Image', left_eye_image)
        cv2.imshow('Right Eye Image', right_eye_image)

        cv2.drawContours(frame, [left_eye_box], 0, BOX_COLOR, BOX_THICKNESS)
        cv2.drawContours(frame, [right_eye_box], 0, BOX_COLOR, BOX_THICKNESS)

    return frame

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_with_boxes = detect_eyes(frame)

        cv2.imshow('Eye Detection', frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()