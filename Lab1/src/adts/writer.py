import serial
import os
import time


class Writer:
    """Handles the writing to a file or to the serial port"""

    def __init__(
        self,
        serial_port: str,
        write_to_serial: bool = False,
    ):
        self.write_to_serial = write_to_serial

        self.logs_file = os.path.normpath("Lab1/text_files/logs.txt")
        # Create the file
        with open(self.logs_file, "w") as f:
            pass

        if write_to_serial:
            self.serial_port = serial.Serial(
                serial_port,
                baudrate=9600,
                bytesize=8,
                timeout=2,
                parity="N",
                xonxoff=0,
                stopbits=serial.STOPBITS_ONE,
            )

        else:
            self.commands_file = os.path.normpath("Lab1/text_files/commands.txt")
            # Create the file
            with open(self.commands_file, "w") as f:
                pass

    def send_command(
        self, command: str, get_starting_point_to_test: str = False
    ) -> str:
        """Sends a command to the serial port or to the command file"""

        if self.write_to_serial:
            self.serial_port.write(bytes(command + "\r", "Ascii"))

            return self.read_and_wait(2)

        else:
            with open(self.commands_file, "a") as f:
                f.write(command + "\r\n")

            # Emulate answers
            if not get_starting_point_to_test:
                return "Done.\r"
            else:
                return "LISTPV cur\r\nPosition CUR\r\n 1:0        2:-3923    3:0        4: 1       5:0       \r\n X: 6508    Y:-353     Z: 8278    P: 231     R:-201    \r\n>"

    def read_and_wait(self, timeout):
        """Read the answer from the serial port"""

        output = ""

        start_time = time.time()
        while True:
            to_read = self.serial_port.in_waiting

            if to_read > 0:
                output += self.serial_port.read(to_read).decode("Ascii")
                break

            if time.time() - start_time > timeout:
                break

        # Logs
        with open(self.logs_file, "a") as f:
            f.write(output)

        return output

    def close(self):
        """Closes the serial port"""

        if self.write_to_serial:
            self.serial_port.close()
