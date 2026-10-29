import cv2
import numpy as np
import turtle
import os

# ==========================================
# SETTINGS
# ==========================================

image_path = "Linkin_Park.png"

# Choose your drawing color (RGB)
LOGO_COLOR = (0, 0, 0) # Red

BACKGROUND_COLOR = "white"
PEN_SIZE = 1

# Animation settings
TURTLE_SPEED = 5
SCREEN_DELAY = 35

# ==========================================
# LOAD IMAGE
# ==========================================

if not os.path.exists(image_path):
    raise FileNotFoundError(
        f"Image '{image_path}' not found. "
        "Place lp.jpeg in the same folder as this Python file."
    )

img = cv2.imread(image_path)

if img is None:
    raise ValueError(f"Could not load image: {image_path}")

HEIGHT, WIDTH = img.shape[:2]

print(f"Image loaded successfully: {WIDTH} x {HEIGHT}")

# ==========================================
# IMAGE PROCESSING
# ==========================================

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the dark logo on the white background
_, mask = cv2.threshold(
    gray,
    150,
    255,
    cv2.THRESH_BINARY_INV
)

# Find the logo outlines, including internal white spaces
contours, hierarchy = cv2.findContours(
    mask,
    cv2.RETR_CCOMP,
    cv2.CHAIN_APPROX_SIMPLE
)

if hierarchy is None or not contours:
    raise ValueError("No logo contours detected. Check your image.")

hierarchy = hierarchy[0]

# ==========================================
# SORT CONTOURS FROM LEFT TO RIGHT
# ==========================================

contour_pairs = list(zip(contours, hierarchy))

# Sort by horizontal position
contour_pairs.sort(
    key=lambda item: cv2.boundingRect(item[0])[0]
)

# ==========================================
# TURTLE SCREEN
# ==========================================

screen = turtle.Screen()

screen.setup(
    width=min(WIDTH + 100, 1800),
    height=min(HEIGHT + 100, 1000)
)

screen.title("LINKIN PARK Logo Drawing")
screen.bgcolor(BACKGROUND_COLOR)
screen.colormode(255)

# Control animation speed
screen.tracer(1, SCREEN_DELAY)

# ==========================================
# TURTLE SETUP
# ==========================================

t = turtle.Turtle()

t.shape("turtle")
t.speed(TURTLE_SPEED)
t.pensize(PEN_SIZE)
t.hideturtle()

# ==========================================
# COORDINATE CONVERSION
# ==========================================

def to_turtle_coords(x, y):
    tx = x - WIDTH / 2
    ty = HEIGHT / 2 - y

    return tx, ty

# ==========================================
# DRAW FILLED CONTOURS
# ==========================================

def draw_contour(cnt, color):
    """Draw and fill one contour using Turtle."""

    if len(cnt) < 3:
        return

    points = cnt[:, 0, :]

    # Move to the starting point
    start_x, start_y = points[0]

    t.penup()
    t.goto(to_turtle_coords(start_x, start_y))

    t.color(color)
    t.pendown()

    t.begin_fill()

    # Draw the outline
    for point in points[1:]:
        x, y = point
        t.goto(to_turtle_coords(x, y))

    # Close the shape
    t.goto(to_turtle_coords(start_x, start_y))

    t.end_fill()
    t.penup()

# ==========================================
# DRAW THE LOGO
# ==========================================

print("Drawing LINKIN PARK logo...")

# First, draw the outside shapes in the selected color
for cnt, h in contour_pairs:

    # Ignore tiny noise
    if cv2.contourArea(cnt) < 10:
        continue

    # Parent contour = -1 means an outer contour
    if h[3] == -1:
        draw_contour(cnt, LOGO_COLOR)

# Next, fill internal spaces with the background color
for cnt, h in contour_pairs:

    if cv2.contourArea(cnt) < 10:
        continue

    # Child contours represent holes inside the logo
    if h[3] != -1:
        draw_contour(cnt, BACKGROUND_COLOR)

# ==========================================
# FINISH
# ==========================================

t.hideturtle()

screen.update()

print("LINKIN PARK logo drawing completed!")

turtle.done()
