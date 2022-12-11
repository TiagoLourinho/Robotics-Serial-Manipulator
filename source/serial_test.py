import serial
from time import time
import json

SERIAL_PORT = "/dev/ttyUSB0"

ser = serial.Serial(
    SERIAL_PORT,
    baudrate=9600,
    bytesize=8,
    timeout=2,
    parity="N",
    xonxoff=0,
    stopbits=serial.STOPBITS_ONE,
)

commands = [
    "DEFP cur",
    "HERE cur",
    "LISTPV cur",
    "DIMP dsadassa[2]",
    "HERE dsadassa[1]",
    "SETPVC dsadassa[1] X 6500",
]

reports = []
for cmd in commands:

    ser.write(bytes(cmd + "\r", "Ascii"))

    start_for_timeout = time()
    start = start_for_timeout

    time_between_writes = []
    bytes_to_read = []
    answers_parts = []

    quit = False
    while not quit:

        while True:
            # Global timeout
            if time() - start_for_timeout > 3:
                quit = True
                break

            to_read = ser.in_waiting

            if to_read:
                break

        if not quit:
            end = time()

            time_between_writes.append(round(end - start, 3))
            bytes_to_read.append(to_read)
            answers_parts.append(ser.read(to_read).decode("Ascii"))

            start = end

    reports.append(
        {
            "command": cmd,
            "total_time": sum(time_between_writes),
            "n_writes": len(bytes_to_read),
            "time_between_writes": time_between_writes,
            "answerpointss_parts": answers_parts,
            "bytes_to_read": bytes_to_read,
        }
    )

json_string = json.dumps(reports, indent=4)

with open("Lab1/text_files/report.json", "w") as f:
    f.write(json_string)
