from random import shuffle
import serial
import time
import os


ser = serial.Serial(
    "COM4" if os.name == "nt" else "/dev/ttyUSB0",
    baudrate=9600,
    bytesize=8,
    timeout=2,
    parity="N",
    xonxoff=0,
    stopbits=serial.STOPBITS_ONE,
)


class Point:
    def __init__(self, cartesian: dict[str:int]):

        self.cartesian = cartesian

    def get_cartesian_coordinates(self) -> dict[str:int]:
        return self.cartesian


def read_and_wait(wait_time):
    """
    This function listens the serial port for wait_time seconds
    waiting for ASCII characters to be sent by the robots
    It returns the string of characters
    """

    output = ""
    flag = True
    start_time = time.time()
    while flag:
        # Wait until there is data waiting in the serial buffer
        if ser.in_waiting > 0:
            # Read data out of the buffer until a carriage return / new line is found
            serString = ser.readline()
            # Print the contents of the serial data
            try:
                output = serString.decode("Ascii")
                print(serString.decode("Ascii"))
            except:
                pass
        else:
            deltat = time.time() - start_time
            if deltat > wait_time:
                flag = False
    return output


def encode_coord_value(value: float):
    return int(value * 10)


def create_vector(points: list[Point]):
    """Creates a vector by following the steps of special notes"""

    # Step 1
    execute_command(f"DIMP points[{len(points)}]")

    for i, point in enumerate(points):

        # Step 1
        execute_command(f"HERE points[{i+1}]")

        # To prevent from choosing a point outside the working zone (step 3)
        error = True
        while error:

            coords = point.get_cartesian_coordinates()
            for coord, value in [(key, coords[key]) for key in shuffle(coords.keys())]:
                if (
                    execute_command(
                        f"SETPVC points[{i+1}] {coord.upper()} {encode_coord_value(value)}"
                    )
                    != "Done."
                ):
                    break
            else:
                error = False

        # Step 4
        execute_command(f"TEACH points[{i+1}]")
        execute_command(f"MOVED points[{i+1}]")  # Move and wait or moved?
        execute_command(f"HERE points[{i+1}]")


def execute_command(command: str):
    ser.write(bytes(command + "\r", "utf-8"))
    time.sleep(0.5)
    return read_and_wait(2)


def main():
    """while True:
    command = input().strip()

    if command == "QUIT":
        break

    execute_command(command)"""

    points = [
        Point({"x": 0, "y": 0, "z": 0}),
        Point({"x": 0, "y": 0, "z": 0}),
        Point({"x": 0, "y": 0, "z": 0}),
    ]

    create_vector(points)

    execute_command(f"MOVES points 1 {len(points)}")

    ser.close()


if __name__ == "__main__":
    main()
