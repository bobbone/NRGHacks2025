from ctypes import *
from time import sleep
user32 = windll.user32
kernel32 = windll.kernel32
delay = 0.01

import win32gui
import win32con

class Key:
        a = 0x41
        b = 0x42
        c = 0x43
        d = 0x44
        e = 0x45
        f = 0x46
        g = 0x47
        h = 0x48
        i = 0x49
        j = 0x4A
        k = 0x4B
        l = 0x4C
        m = 0x4D
        n = 0x4E
        o = 0x4F
        p = 0x50
        q = 0x51
        r = 0x52
        s = 0x53
        t = 0x54
        u = 0x55
        v = 0x56
        w = 0x57
        x = 0x58
        y = 0x59
        z = 0x5A
        SPACE = 0x20

WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101

class Keyboard:
        def __init__(self):
                self.Key = Key()
        def press(self, key):
                """Presses key"""
                user32.keybd_event(key, 0, 0, 0)
                sleep(delay)
                user32.keybd_event(key, 0, 2, 0)
                sleep(delay)
        def hold(self, key):
                """Holds a key"""
                user32.keybd_event(key, 0, 0, 0)
                sleep(delay)

        def release(self, key):
                """Releases a key"""
                user32.keybd_event(key, 0, 2, 0)
                sleep(delay)

class Point(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

class Mouse:
        def __init__(self):
                self.left = [0x0002, 0x0004]
                self.right = [0x0008, 0x00010]
                self.middle = [0x00020, 0x00040]

        def move(self, x, y):
                """Moves the cursor"""
                pt = Point()
                user32.GetCursorPos(byref(pt))
                user32.SetCursorPos(x+pt.x, y+pt.y)
        
        def set(self, x, y):
                user32.SetCursorPos(x, y)

        def click(self, button):
                """Clicks button"""
                user32.mouse_event(button[0], 0, 0, 0, 0)
                sleep(delay)
                user32.mouse_event(button[1], 0, 0, 0, 0)
                sleep(delay)

        def holdclick(self, button):
                """Start pressing button"""
                user32.mouse_event(button[0], 0, 0, 0, 0)
                sleep(delay)

        def releaseclick(self, button):
                """Release button"""
                user32.mouse_event(button[1])
                sleep(delay)
