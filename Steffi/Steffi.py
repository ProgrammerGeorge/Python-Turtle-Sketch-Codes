import cv2
import turtle
import numpy as np
import time
import os

# ==========================================
# SETTINGS
# ==========================================

image = "Steffi.png"

PEN_COLOR = "gray20"
PEN_SIZE = 2

# Drawing speed
POINT_DELAY = 0.005
CONTOUR_DELAY = 0.3

# ==========================================
# LOAD IMAGE
# ==========================================

if not os.path.exists(image):
    raise FileNotFoundError(
        f"Image '{image}' not found. "
        "Place the image in the same folder as this Python file "
        "or provide its full path."
    )

img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError(f"Could not read image: {image}")

HEIGHT, WIDTH = img.shape

print(f"Image loaded: {image}")
print(f"Image dimensions: {WIDTH} x {HEIGHT}")

# ==========================================
# PROCESS IMAGE
# ==========================================

blurred = cv2.GaussianBlur(img, (5, 5), 0)

binary = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    3
)

# ==========================================
# FIND CONTOURS
# ==========================================

contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_NONE
)

if hierarchy is None or len(contours) == 0:
    raise ValueError(
        "No contours detected. Try adjusting the threshold settings."
    )

# ==========================================
# FILTER AND SORT CONTOURS
# ==========================================

drawing_contours = []

for i, contour in enumerate(contours):

    x, y, w, h = cv2.boundingRect(contour)

    # Ignore tiny contours caused by image noise
    if cv2.arcLength(contour, False) < 10:
        continue

    drawing_contours.append({
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "contour": contour,
        "index": i
    })

# Sort from left to right, then top to bottom
drawing_contours.sort(
    key=lambda c: (c["x"], c["y"])
)

# ==========================================
# CREATE TURTLE SCREEN
# ==========================================

screen = turtle.Screen()

screen.setup(
    width=WIDTH + 100,
    height=HEIGHT + 100
)

screen.bgcolor("white")
screen.title("Steffi Portrait - Slow Turtle Drawing")

# Enable manual screen updates
screen.tracer(0, 0)

t = turtle.Turtle()

t.hideturtle()
t.speed(0)
t.pencolor(PEN_COLOR)
t.pensize(PEN_SIZE)
t.penup()

# ==========================================
# CONVERT IMAGE COORDINATES
# ==========================================

def turtle_position(x, y):

    tx = x - WIDTH / 2
    ty = HEIGHT / 2 - y

    return tx, ty

# ==========================================
# DRAW EACH CONTOUR SLOWLY
# ==========================================

def draw_contour(contour):

    points = contour.reshape(-1, 2)

    if len(points) < 2:
        return

    # Start at the leftmost point of this contour
    start_index = np.argmin(points[:, 0])

    points = np.roll(
        points,
        -start_index,
        axis=0
    )

    # Move to the starting point without drawing
    x, y = points[0]

    tx, ty = turtle_position(x, y)

    t.penup()
    t.goto(tx, ty)
    t.pendown()

    # Trace the contour point by point
    for x, y in points:

        tx, ty = turtle_position(x, y)

        t.goto(tx, ty)

        # Make the drawing visible
        screen.update()

        if POINT_DELAY > 0:
            time.sleep(POINT_DELAY)

    t.penup()

# ==========================================
# DRAW THE PORTRAIT
# ==========================================

print("Starting slow portrait drawing...")

start_time = time.time()

total = len(drawing_contours)

for i, item in enumerate(drawing_contours, start=1):

    draw_contour(item["contour"])

    print(
        f"Progress: {i}/{total} contours",
        end="\r",
        flush=True
    )

    # Pause before starting the next contour
    if CONTOUR_DELAY > 0:
        time.sleep(CONTOUR_DELAY)

# ==========================================
# FINISH
# ==========================================

screen.update()

elapsed = time.time() - start_time

print("\nPortrait drawing completed!")
print(f"Total contours: {total}")
print(f"Time taken: {elapsed:.2f} seconds")

turtle.done()
