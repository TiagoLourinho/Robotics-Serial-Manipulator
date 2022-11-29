import serial
import os
import time


class Writer:
    """Handles the writing to a file or to the serial port"""

    def __init__(
        self,
        serial_port: str,
        write_to_serial: bool = False,
        sleeping_time: float = 0.5,
    ):
        self.write_to_serial = write_to_serial
        self.sleeping_time = sleeping_time

        if write_to_serial:
            self.__serial = serial.Serial(
                serial_port,
                baudrate=9600,
                bytesize=8,
                timeout=2,
                parity="N",
                xonxoff=0,
                stopbits=serial.STOPBITS_ONE,
            )

        else:
            self.__file = os.path.normpath("Lab1/commands.txt")

            # Create the file
            with open(self.__file, "w") as f:
                pass

    def send_command(self, command: str) -> str:
        """Sends a command to the serial port or to the command file"""

        if self.write_to_serial:
            self.__serial.write(bytes(command + "\r", "Ascii"))
            time.sleep(self.sleeping_time)

            return self.read_and_wait(2)

        else:
            with open(self.__file, "a") as f:
                f.write(command + "\r\n")

            return "Done.\r"

    def read_and_wait(self, wait_time):
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
            if self.__serial.in_waiting > 0:
                # Read data out of the buffer until a carriage return / new line is found
                serString = self.__serial.readline()
                # Print the contents of the serial data
                try:
                    output += serString.decode("Ascii")
                    print(serString.decode("Ascii"))
                except:
                    pass
            else:
                deltat = time.time() - start_time
                if deltat > wait_time:
                    flag = False
        return output

    def close(self):
        """Closes the serial port"""

        if self.write_to_serial:
            self.__serial.close()
