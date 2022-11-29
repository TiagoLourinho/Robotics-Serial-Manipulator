class Point:
    """A point represented in cartesian coordinates"""

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Point({round(self.x,2)},{round(self.y,2)},{round(self.z,2)})"

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, num: float) -> "Point":
        return Point(self.x * num, self.y * num, self.z * num)

    def __rmul__(self, num: float) -> "Point":
        return Point(self.x * num, self.y * num, self.z * num)
