import sys
import os
import cv2
import numpy as np
import pygame
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QTimer
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import mediapipe as mp
from datetime import datetime

# Настройки
DRAWING_IMAGES = ['kitty.jpg']  # Указать свои пути
SAVE_DIR = 'gallery'
os.makedirs(SAVE_DIR, exist_ok=True)

# Mediapipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

# --------- PyQt Меню ---------
class MenuWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AR App Menu")
        self.setStyleSheet("background-color: #2C3E50; color: white; font-size: 20px;")
        self.setGeometry(100, 100, 600, 500)  # Увеличим размер окна меню
        layout = QVBoxLayout()

        self.btn_drawing = QPushButton("Drawing")
        self.btn_gallery = QPushButton("Gallery")
        self.btn_3d = QPushButton("3D Figures")
        self.btn_drawing.setStyleSheet("background-color: #FF6347; padding: 10px; margin: 10px; border-radius: 10px;")
        self.btn_gallery.setStyleSheet("background-color: #FF6347; padding: 10px; margin: 10px; border-radius: 10px;")
        self.btn_3d.setStyleSheet("background-color: #FF6347; padding: 10px; margin: 10px; border-radius: 10px;")

        layout.addWidget(self.btn_drawing)
        layout.addWidget(self.btn_gallery)
        layout.addWidget(self.btn_3d)
        self.setLayout(layout)

        self.btn_drawing.clicked.connect(self.open_drawing)
        self.btn_gallery.clicked.connect(self.open_gallery)
        self.btn_3d.clicked.connect(self.open_3d)

    def open_drawing(self):
        self.close()
        run_drawing_mode()

    def open_gallery(self):
        self.close()
        run_gallery_mode()

    def open_3d(self):
        self.close()
        run_3d_mode()

# --------- Возвращение в меню ---------
def go_back_to_menu():
    app = QApplication(sys.argv)
    menu = MenuWindow()
    menu.show()
    sys.exit(app.exec_())

# --------- Раскраска ---------
def run_drawing_mode():
    cap = cv2.VideoCapture(0)
    pygame.init()
    WIDTH, HEIGHT = 960, 720
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drawing Mode")

    background_img = None
    if DRAWING_IMAGES:
        try:
            img_path = DRAWING_IMAGES[0]
            background_img = pygame.image.load(img_path)
            background_img = pygame.transform.scale(background_img, (WIDTH, HEIGHT))
        except:
            print(f"Could not load image: {DRAWING_IMAGES[0]}")

    canvas = pygame.Surface((WIDTH, HEIGHT))
    if background_img:
        canvas.blit(background_img, (0, 0))
    else:
        canvas.fill((255, 255, 255))

    draw_color = (255, 0, 0)
    thickness = 5
    changing_thickness = False

    clock = pygame.time.Clock()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        face_result = face_detection.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm = hand_landmarks.landmark
                fingers = [lm[8].y < lm[6].y, lm[12].y < lm[10].y, lm[16].y < lm[14].y, lm[20].y < lm[18].y]
                finger_count = fingers.count(True)

                x = int(lm[8].x * WIDTH)
                y = int(lm[8].y * HEIGHT)

                if fingers[0] and not any(fingers[1:]):  # Только указательный
                    pygame.draw.circle(canvas, draw_color, (x, y), thickness)

                if finger_count == 2:
                    draw_color = (0, 255, 0)
                elif finger_count == 3:
                    draw_color = (0, 0, 255)
                elif finger_count == 5:
                    draw_color = (255, 0, 0)
                elif finger_count == 4:
                    thickness = max(thickness - 1, 1)
                elif finger_count == 0:
                    changing_thickness = True
                else:
                    changing_thickness = False

        if changing_thickness:
            thickness = min(thickness + 1, 50)

        if face_result.detections:
            for detection in face_result.detections:
                mp_draw.draw_detection(frame, detection)

        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = os.path.join(SAVE_DIR, f'drawing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            pygame.image.save(canvas, filename)
        elif key == ord('u'):  # Кнопка для стирания
            canvas.fill((255, 255, 255))  # Очистить холст

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                cap.release()
                cv2.destroyAllWindows()
                return
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:  # Возвращение в меню
                    pygame.quit()
                    go_back_to_menu()

        screen.fill((200, 200, 200))
        screen.blit(canvas, (0, 0))

        pygame.draw.rect(screen, (0, 0, 0), (WIDTH - 30, 50, 20, 200), 2)
        pygame.draw.rect(screen, draw_color, (WIDTH - 28, 250 - thickness * 4, 16, thickness * 4))

        font = pygame.font.SysFont(None, 24)
        text = font.render(f"Color: {draw_color} | Thickness: {thickness}", True, (0, 0, 0))
        screen.blit(text, (10, 10))

        pygame.display.update()
        clock.tick(60)

# --------- Галерея ---------
def run_gallery_mode():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Gallery")
    images = [pygame.image.load(os.path.join(SAVE_DIR, f)) for f in os.listdir(SAVE_DIR)]
    index = 0

    while True:
        screen.fill((0, 0, 0))
        if images:
            img = pygame.transform.scale(images[index], (640, 480))
            screen.blit(img, (0, 0))
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            elif event.type == KEYDOWN:
                if event.key == K_RIGHT:
                    index = (index + 1) % len(images)
                elif event.key == K_LEFT:
                    index = (index - 1) % len(images)
                elif event.key == K_ESCAPE:  # Возвращение в меню
                    pygame.quit()
                    go_back_to_menu()

        pygame.display.update()
# --------- 3D Режим ---------
def run_3d_mode():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D AR Drawing")

    glutInit()
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    glEnable(GL_DEPTH_TEST)

    figures = ['cube', 'pyramid', 'sphere']
    current_figure = 0
    draw_points = []
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    current_color = colors[0]

    cap = cv2.VideoCapture(0)
    running = True

    def draw_cube():
        vertices = [(1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
                    (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)]
        edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,7),(7,6),(6,4),
                 (0,4),(1,5),(2,7),(3,6)]
        glBegin(GL_LINES)
        glColor3fv(current_color)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

    def draw_pyramid():
        vertices = [(0, 1, 0), (-1, -1, 1), (1, -1, 1), (1, -1, -1), (-1, -1, -1)]
        edges = [(0,1),(0,2),(0,3),(0,4), (1,2),(2,3),(3,4),(4,1)]
        glBegin(GL_LINES)
        glColor3fv(current_color)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

    def draw_sphere():
        glColor3fv(current_color)
        quadric = gluNewQuadric()
        gluSphere(quadric, 1, 16, 16)

    def draw_current_figure():
        if figures[current_figure] == 'cube': draw_cube()
        elif figures[current_figure] == 'pyramid': draw_pyramid()
        elif figures[current_figure] == 'sphere': draw_sphere()

    def handle_gestures(result):
        nonlocal current_color, current_figure, draw_points
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
            fingers = [lm[8].y < lm[6].y, lm[12].y < lm[10].y, lm[16].y < lm[14].y, lm[20].y < lm[18].y]
            finger_count = fingers.count(True)
            if fingers[0] and not any(fingers[1:]):
                x = (lm[8].x - 0.5) * 4
                y = -(lm[8].y - 0.5) * 3
                z = -3.0
                draw_points.append((x, y, z))
            if finger_count == 0:
                current_figure = (current_figure + 1) % len(figures)
            if finger_count == 4:
                draw_points = []
            if 0 < finger_count <= len(colors):
                current_color = colors[finger_count - 1]

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        face_result = face_detection.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if face_result.detections:
            for detection in face_result.detections:
                mp_draw.draw_detection(frame, detection)
        handle_gestures(result)

        cv2.imshow("Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glRotatef(1, 0, 1, 0)
        draw_current_figure()
        glBegin(GL_POINTS)
        glColor3fv(current_color)
        for p in draw_points:
            glVertex3fv(p)
        glEnd()
        pygame.display.flip()
        pygame.time.wait(10)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

# --------- Запуск ---------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    menu = MenuWindow()
    menu.show()
    sys.exit(app.exec_())
