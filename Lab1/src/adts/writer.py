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

            return self.read_and_wait()

        else:
            with open(self.commands_file, "a") as f:
                f.write(command + "\r\n")

            # Emulate answers
            if not get_starting_point_to_test:
                return "Done.\r"
            else:
                # Dummy response in case of not being connected to serial port
                return "LISTPV cur\r\nPosition CUR\r\n 1: 3004        2:-1140    3:-8085        4: 2893       5:-811       \r\n X: 7194    Y: 1134     Z: 4719    P:-103     R:-277    \r\n>"

    def read_and_wait(self, time_until_first_write=0.01, time_between_writes=0.3):
        """Read the answer from the serial port"""

        output = ""

        time.sleep(time_until_first_write)

        start_time = time.time()
        while True:
            to_read = self.serial_port.in_waiting

            if to_read > 0:
                output += self.serial_port.read(to_read).decode("Ascii")
                start_time = time.time()

            if time.time() - start_time > time_between_writes:
                break

        # Logs
        with open(self.logs_file, "a") as f:
            f.write(output)

        return output

    def close(self):
        """Closes the serial port"""

        if self.write_to_serial:
            self.serial_port.close()
