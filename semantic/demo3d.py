import numpy as np
import pyglet
from pyglet.gl import *

# =========================
# Basic math
# =========================
def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n

def offaxis_projection(eye, pa, pb, pc, near=0.1, far=100.0):
    """
    eye: observer position
    pa, pb, pc: screen corners
      pa = lower-left
      pb = lower-right
      pc = upper-left
    """
    vr = normalize(pb - pa)
    vu = normalize(pc - pa)
    vn = normalize(np.cross(vr, vu))

    va = pa - eye
    vb = pb - eye
    vc = pc - eye

    d = -np.dot(va, vn)
    if d <= 1e-5:
        d = 1e-5

    l = np.dot(vr, va) * near / d
    r = np.dot(vr, vb) * near / d
    b = np.dot(vu, va) * near / d
    t = np.dot(vu, vc) * near / d

    proj = np.array([
        [2 * near / (r - l), 0, (r + l) / (r - l), 0],
        [0, 2 * near / (t - b), (t + b) / (t - b), 0],
        [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0]
    ], dtype=np.float32)

    # world -> screen basis
    R = np.eye(4, dtype=np.float32)
    R[0, :3] = vr
    R[1, :3] = vu
    R[2, :3] = vn

    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye

    view = R @ T
    return proj, view

def to_gl_matrix(m):
    return (GLfloat * 16)(*m.T.flatten())

# =========================
# Global scene setup
# =========================

# 折角外侧观察者位置：x>0, z>0
eye = np.array([0.75, 0.08, 0.75], dtype=np.float32)

cube_angle = 0.0

# 两块屏幕真实尺寸（世界坐标）
screen_w = 1.2
screen_h = 0.9

# ---------------------------------------------------
# Screen A: front plane, z = 0
# 它和 Screen B 共用竖直边：x=0, z=0
# viewer 在 z>0 一侧
# ---------------------------------------------------
A_pa = np.array([-screen_w, -screen_h/2, 0.0], dtype=np.float32)   # lower-left
A_pb = np.array([0.0,       -screen_h/2, 0.0], dtype=np.float32)   # lower-right
A_pc = np.array([-screen_w,  screen_h/2, 0.0], dtype=np.float32)   # upper-left

# ---------------------------------------------------
# Screen B: side plane, x = 0
# viewer 在 x>0 一侧
# ---------------------------------------------------
B_pa = np.array([0.0, -screen_h/2,  0.0], dtype=np.float32)        # lower-left
B_pb = np.array([0.0, -screen_h/2, -screen_w], dtype=np.float32)   # lower-right
B_pc = np.array([0.0,  screen_h/2,  0.0], dtype=np.float32)        # upper-left

# cube 放在折角内部空间（x<0, z<0）
cube_center = np.array([-0.22, 0.0, -0.22], dtype=np.float32)

# =========================
# Drawing helpers
# =========================
def draw_axes():
    glBegin(GL_LINES)
    glColor3f(1, 0, 0)  # x
    glVertex3f(0, 0, 0)
    glVertex3f(0.4, 0, 0)

    glColor3f(0, 1, 0)  # y
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0.4, 0)

    glColor3f(0, 0, 1)  # z
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 0.4)
    glEnd()

def draw_floor_grid(size=1.2, step=0.1):
    glColor3f(0.22, 0.22, 0.25)
    glBegin(GL_LINES)
    y = -0.28
    xs = np.arange(-size, 0.001, step)
    zs = np.arange(-size, 0.001, step)

    for x in xs:
        glVertex3f(x, y, -size)
        glVertex3f(x, y, 0.0)

    for z in zs:
        glVertex3f(-size, y, z)
        glVertex3f(0.0, y, z)
    glEnd()

def draw_corner_edges():
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glColor3f(0.8, 0.8, 0.8)
    # shared vertical edge
    glVertex3f(0, -screen_h/2, 0)
    glVertex3f(0,  screen_h/2, 0)
    glEnd()
    glLineWidth(1.0)

def draw_cube(size=0.28):
    s = size / 2.0
    glBegin(GL_QUADS)

    # front face (+z)
    glColor3f(0.95, 0.35, 0.35)
    glVertex3f(-s, -s,  s)
    glVertex3f( s, -s,  s)
    glVertex3f( s,  s,  s)
    glVertex3f(-s,  s,  s)

    # back face (-z)
    glColor3f(0.35, 0.95, 0.35)
    glVertex3f(-s, -s, -s)
    glVertex3f(-s,  s, -s)
    glVertex3f( s,  s, -s)
    glVertex3f( s, -s, -s)

    # left face (-x)
    glColor3f(0.35, 0.35, 0.95)
    glVertex3f(-s, -s, -s)
    glVertex3f(-s, -s,  s)
    glVertex3f(-s,  s,  s)
    glVertex3f(-s,  s, -s)

    # right face (+x)
    glColor3f(0.95, 0.95, 0.35)
    glVertex3f( s, -s, -s)
    glVertex3f( s,  s, -s)
    glVertex3f( s,  s,  s)
    glVertex3f( s, -s,  s)

    # top
    glColor3f(0.35, 0.95, 0.95)
    glVertex3f(-s,  s, -s)
    glVertex3f(-s,  s,  s)
    glVertex3f( s,  s,  s)
    glVertex3f( s,  s, -s)

    # bottom
    glColor3f(0.85, 0.3, 0.85)
    glVertex3f(-s, -s, -s)
    glVertex3f( s, -s, -s)
    glVertex3f( s, -s,  s)
    glVertex3f(-s, -s,  s)

    glEnd()

def draw_scene():
    global cube_angle

    draw_axes()
    draw_floor_grid()
    draw_corner_edges()

    glPushMatrix()
    glTranslatef(cube_center[0], cube_center[1], cube_center[2])
    glRotatef(cube_angle, 0.8, 1.0, 0.2)
    draw_cube()
    glPopMatrix()

# =========================
# Window
# =========================
class ScreenWindow(pyglet.window.Window):
    def __init__(self, title, pa, pb, pc, x, y, width=820, height=620):
        super().__init__(width=width, height=height, caption=title, resizable=False)
        self.pa = pa
        self.pb = pb
        self.pc = pc
        self.set_location(x, y)

        glEnable(GL_DEPTH_TEST)
        glClearColor(0.03, 0.03, 0.06, 1.0)

    def on_draw(self):
        self.clear()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        proj, view = offaxis_projection(eye, self.pa, self.pb, self.pc, near=0.1, far=30.0)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glLoadMatrixf(to_gl_matrix(proj))

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glLoadMatrixf(to_gl_matrix(view))

        draw_scene()

    def on_key_press(self, symbol, modifiers):
        global eye
        step = 0.04

        if symbol == pyglet.window.key.W:
            eye[2] -= step
        elif symbol == pyglet.window.key.S:
            eye[2] += step
        elif symbol == pyglet.window.key.A:
            eye[0] -= step
        elif symbol == pyglet.window.key.D:
            eye[0] += step
        elif symbol == pyglet.window.key.Q:
            eye[1] += step
        elif symbol == pyglet.window.key.E:
            eye[1] -= step

        print("eye =", eye)

# =========================
# App
# =========================
win_a = ScreenWindow("Screen A (Front Plane)", A_pa, A_pb, A_pc, x=40,  y=60)
win_b = ScreenWindow("Screen B (Side Plane)",  B_pa, B_pb, B_pc, x=900, y=60)

def update(dt):
    global cube_angle
    cube_angle += 28.0 * dt

pyglet.clock.schedule_interval(update, 1/60.0)
pyglet.app.run()