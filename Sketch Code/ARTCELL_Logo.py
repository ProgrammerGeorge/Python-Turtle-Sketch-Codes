import cv2
import numpy as np
import turtle
image_path = "ARTCELL.png"


img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not load image at '{image_path}'")

HEIGHT, WIDTH, _ = img.shape


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)


contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
)

screen = turtle.Screen()
screen.setup(width=WIDTH + 50, height=HEIGHT + 50)
screen.bgcolor("white")
screen.colormode(255)

screen.tracer(1)
screen.delay(8)

t = turtle.Turtle()
t.shape("turtle")
t.speed(5)
t.pensize(2)


RED_COLOR = (227, 6, 19)
t.color(RED_COLOR, RED_COLOR)



def to_turtle_coords(x, y):
    tx = x - (WIDTH / 2)
    ty = (HEIGHT / 2) - y
    return tx, ty



if contours and hierarchy is not None:
    hierarchy_list = hierarchy[0]


    contour_pairs = list(zip(contours, hierarchy_list))


    contour_pairs.sort(key=lambda pair: cv2.boundingRect(pair[0])[0])


    for cnt, h in contour_pairs:
        if cv2.contourArea(cnt) < 10:
            continue


        if h[3] == -1:
            t.penup()
            start_x, start_y = cnt[0][0]
            t.goto(to_turtle_coords(start_x, start_y))
            t.pendown()

            t.begin_fill()
            for pt in cnt[1:]:
                x, y = pt[0]
                t.goto(to_turtle_coords(x, y))
            t.goto(to_turtle_coords(start_x, start_y))
            t.end_fill()


    t.color("white", "white")
    for cnt, h in contour_pairs:
        if cv2.contourArea(cnt) < 10:
            continue


        if h[3] != -1:
            t.penup()
            start_x, start_y = cnt[0][0]
            t.goto(to_turtle_coords(start_x, start_y))
            t.pendown()

            t.begin_fill()
            for pt in cnt[1:]:
                x, y = pt[0]
                t.goto(to_turtle_coords(x, y))
            t.goto(to_turtle_coords(start_x, start_y))
            t.end_fill()

t.hideturtle()
turtle.done()
