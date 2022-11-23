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


def encode_angles(angles: dict[int:int]):
    encoder = {
        1: {"angle": 250, "min": -31960, "max": 31950},
        2: {"angle": 170, "min": -16960, "max": 25972},
        3: {"angle": 225, "min": -28480, "max": 28942},
        4: {"angle": 180, "min": -27048, "max": 28133},
        5: {"angle": 360, "min": -31929, "max": 31956},
    }

    return {
        i: int(
            encoder[i]["min"]
            + (encoder[i]["max"] - encoder[i]["min"])
            * (angles[i] / encoder[i]["angle"])
        )
        for i in angles.keys()
    }


def encode_coords(coords: dict[str:int]):
    return {key: int(item * 10) for key, item in coords.items()}


def create_vector(points: list[dict[str:int]]):

    execute_command(f"DIMP points[{len(points)}]")

    for i, point in enumerate(points):
        execute_command(f"HERE points[{i+1}]")

        for coord, value in point.items():
            execute_command(f"SETPVC points[{i+1}] {coord} {value}")


def read_and_wait(wait_time):
    """
    This function listens the serial port for wait_time seconds
    waiting for ASCII characters to be sent by the robot
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


def execute_command(command: str):
    ser.write(bytes(command + "\r", "utf-8"))
    time.sleep(0.5)
    read_and_wait(2)


def main():
    while True:
        command = input().strip()

        if command == "QUIT":
            break

        execute_command(command)

    ser.close()


if __name__ == "__main__":
    main()
