from random import shuffle

from adts import Point, Writer


class Robot:
    """Scorbot ER-7 serial manipulator"""

    def __init__(self, writer: Writer, speed: int):
        self.writer = writer
        writer.send_command(f"SPEED {speed}")

    def get_starting_point(self) -> Point:
        """Retrieves the initial point from the robot"""

        self.writer.send_command("DEFP cur")
        self.writer.send_command("HERE cur")
        tokens = self.writer.send_command("LISTPV cur")

        # See example of the response got from the robot in README.md
        tokens = tokens[tokens.index("1:") :].replace(": ", ":").split()[:-1]

        # Writing to a file (the return is always Done.)
        if len(tokens) == 1:
            return Point(0, 0, 0)

        # Extract the values from the following string (after spliting):
        # 1:0     2:1791  3:2746   4:0      5:-1
        # X:1690  Y:0	  Z:6011   P:-636   R:-1
        coords = {
            key: int(tokens[i][tokens[i].index(":") + 1 :])
            for i, key in enumerate([1, 2, 3, 4, 5, "X", "Y", "Z", "P", "R"])
        }

        return Point(*(self.decode_cartesian(coords[c]) for c in ["X", "Y", "Z"]))

    def create_vector_of_points(self, points: list[Point]):
        """Creates a vector of points by following the steps in the special notes of the lab slides"""

        # Step 1.1
        self.writer.send_command(f"DIMP points[{len(points)}]")

        for i, point in enumerate(points):

            # Step 1.2
            self.writer.send_command(f"HERE points[{i+1}]")

            # Step 3 (preventing the creation of a temporary point outside the working zone)
            enc_coords = self.get_encoded_cartesian_coordinates(point)
            keys = list(enc_coords.keys())

            error = True
            while error:
                for coord, enc_value in [(key, enc_coords[key]) for key in keys]:

                    # Step 1.3
                    ans = self.writer.send_command(
                        f"SETPVC points[{i+1}] {coord} {enc_value}"
                    )

                    if self.is_error(ans):
                        shuffle(keys)
                        break
                else:
                    error = False

            """ 
            # Step 4
            self.writer.send_command(f"TEACH points[{i+1}]")
            self.writer.send_command(f"MOVED points[{i+1}]")  
            self.writer.send_command(f"HERE points[{i+1}]") 
            """

    def is_error(self, ans: str):
        """Checks whether the response of the robot represents an error"""

        return not "Done." in ans

    def get_encoded_cartesian_coordinates(self, point: Point) -> dict[str:int]:
        """Encodes and returns the cartesian coordinates"""

        return {
            "X": self.encode_cartesian(point.x),
            "Y": self.encode_cartesian(point.y),
            "Z": self.encode_cartesian(point.z),
        }

    def encode_cartesian(self, value: float) -> int:
        """Encodes a cartesian coordinate"""

        return int(value * 10)

    def decode_cartesian(self, value: float) -> int:
        """Decodes a cartesian coordinate"""

        return int(value / 10)
