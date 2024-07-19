from collections import namedtuple
import math

Coord = namedtuple("Coord", ["q", "r"])

def naivec2i(c, rad):
    q = c.q + rad
    r = c.r + rad
    coords_above = 0
    for rprime in range(r):
        coords_above += 2*rad+1 - abs(rad-rprime)
    q_start = -c.r if r < rad else c.r
    return coords_above + (q - q_start)

def fastc2i(c, rad):
    q = c.q + rad
    r = c.r + rad
    fh = min(r, rad)
    coords_above = fh*(rad+1) + fh*(fh-1)//2
    if fh < r:
        d = r - fh
        coords_above += d*(2*rad+1) - d*(d-1)//2
    q_start = r if r < rad else -rad
    print(coords_above, q, q_start, q)
    return coords_above + (c.q - q_start)

print(naivec2i(Coord(-512, 1), 512))
q, r = 0, -2
print(naivec2i(Coord(q, r), 2), fastc2i(Coord(q, r), 2))