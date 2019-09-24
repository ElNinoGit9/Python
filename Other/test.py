import numpy as np

lat = np.pi/2.
r_earth = 6.3781e6
r_north = 1000
r_n = 1000
deg = np.arccos(r_n/r_earth) * 180 / np.pi
deg_new = deg - r_north/(2*np.pi*r_earth) * 360
deg_H = np.floor(deg_new)
deg_M = np.floor((deg_new - deg_H)*60)
deg_S = np.floor(((deg_new - deg_H)*60 - deg_M)*60)

print(deg_H, ':', deg_M, ':', deg_S)

from win32com.client import Dispatch

speak = Dispatch("SAPI.SpVoice")

speak.Speak("Hello world")
