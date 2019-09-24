from phue import Bridge
import numpy as np
import time

b = Bridge('192.168.1.178')

b.connect()

b.get_api

print(b.get_light(1, 'on'))

a = b.get_light(1, 'xy')

print(a)
b.set_light(1, 'xy', [0.1, 0.1])

print(b.lights)

for x in np.linspace(0,1,1):
    for y in np.linspace(0,1,1):
        b.set_light(1, 'xy', [x, y])
