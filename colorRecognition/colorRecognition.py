import cv2

# Loading camera
cap = cv2.VideoCapture(0)

# Setting the resolution of the display frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# Finding the color
def color_mapping(hue_value):
    # Map colors to their range
    color_ranges = {
        'Red': [0, 10],
        'Orange': [11, 20],
        'Yellow': [21, 30],
        'Green-yellow': [31, 50],
        'Green': [51, 85],
        'Turquoise': [86, 120],
        'Blue': [121, 160],
        'Purple': [161, 179],
        'Pink': [150, 170],
        'Brown': [10, 30],    # Overlapping range w orange and yellow
        'Grey': [0, 179],   # All hues
        'White': [0, 179],    # All hues
        'Black': [0, 179],    # All hues
        'Light-blue': [100, 140],
        'Beige': [20, 40]    # Overlapping range w brown and yellow
    }

    for color, (lower, upper) in color_ranges.items():
        if lower <= hue_value <= upper:
            return color

    return 'Unknown'


while True:
    # Get a frame, .read() returns a boolean value and the captured frame
    _, bgr_frame = cap.read()

    # BGR to HSV
    hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

    height, width, _ = hsv_frame.shape    # .shape returns height, width, channel(color image - 3, greyscale - 2)

    # Calculate the center coordinates
    cx = int(width / 2)
    cy = int(height / 2)

    # Find the center pixel
    center_pixel_hsv = hsv_frame[cy, cx]    # height, width

    # Finding 'hue' since it's in HSV
    hue_value = int(center_pixel_hsv[0])

    # What color is hue?
    color = color_mapping(hue_value)
    # print(color)

    # Color the text the same color as the color
    center_pixel_bgr = bgr_frame[cy, cx]
    b, g, r = int(center_pixel_bgr[0]), int(center_pixel_bgr[1]), int(center_pixel_bgr[2])

    # Draw a rectangle to act as background for the text
    cv2.rectangle(bgr_frame, (cx - 220, 10), (cx + 200, 120), (255, 255, 255), -1)

    # Color's name as text
    cv2.putText(bgr_frame, color, (cx - 200, 100), 0, 2, (b, g, r), 5)

    # Draw a circle at the center
    cv2.circle(bgr_frame, (cx, cy), 5, (25, 25, 25), 3)

    cv2.imshow("Frame", bgr_frame)

    # If you press a key the input will be stored onto 'key'
    key = cv2.waitKey(1)  # waitKey(1) - Wait's 1ms and if we do nothing, it moves to the next frame

    # If you hit 's' the program stops, s = 27
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()