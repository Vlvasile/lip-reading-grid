import cv2

LIP_POINTS = [61, 291, 0, 17]
MOUTH_WIDTH = 80
MOUTH_HEIGHT = 40
SCALE = 2.0


def get_mouth_center(landmarks, width, height):
    xs = []
    ys = []

    for index in LIP_POINTS:
        point = landmarks[index]
        xs.append(point.x * width)
        ys.append(point.y * height)

    center_x = int(sum(xs) / len(xs))
    center_y = int(sum(ys) / len(ys))
    return center_x, center_y


def get_mouth_box(landmarks, width, height):
    left_corner = landmarks[61]
    right_corner = landmarks[291]
    mouth_size = (right_corner.x - left_corner.x) * width

    crop_width = int(mouth_size * SCALE)
    crop_height = crop_width // 2

    center_x, center_y = get_mouth_center(landmarks, width, height)

    left = center_x - crop_width // 2
    top = center_y - crop_height // 2
    right = left + crop_width
    bottom = top + crop_height
    return left, top, right, bottom


def crop_mouth(rgb_frame, landmarks):
    height = rgb_frame.shape[0]
    width = rgb_frame.shape[1]

    left, top, right, bottom = get_mouth_box(landmarks, width, height)

    if left < 0 or top < 0 or right > width or bottom > height:
        return None

    mouth = rgb_frame[top:bottom, left:right]
    mouth = cv2.resize(mouth, (MOUTH_WIDTH, MOUTH_HEIGHT))
    return mouth