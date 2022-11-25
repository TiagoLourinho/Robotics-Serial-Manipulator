class Point:
    """A point represented in cartesian coordinates"""

    def __init__(self, X: float = 0, Y: float = 0, Z: float = 0, encoded: bool = False):
        """
        Arguments
        ---------
        X, Y, Z: float
            The x, y and z coordinates in milimeters
        encoded: bool
           Whether or not the given coordinates are already encoded
        """

        self.__X = X if not encoded else self.decode(X)
        self.__Y = Y if not encoded else self.decode(Y)
        self.__Z = Z if not encoded else self.decode(Z)

    def get_encoded_cartesian_coordinates(self) -> dict[str:int]:
        """Encodes and returns the cartesian coordinates"""

        return {
            "X": self.encode(self.__X),
            "Y": self.encode(self.__Y),
            "Z": self.encode(self.__Z),
        }

    def encode(self, value: float) -> int:
        return int(value * 10)

    def decode(self, value: float) -> int:
        return int(value / 10)

    def transform(self, new_origin: "Point"):
        """Transforms the point given a new origin point (translation)"""

        self.__X += new_origin.__X
        self.__Y += new_origin.__Y
        self.__Z += new_origin.__Z
