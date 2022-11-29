# Robotics
Repository for the Robotics course including laboratories 1, 2 and paper related.

## Steps to enable serial port in Linux:
1. `ls /dev/ttyU*`
2. `sudo chmod o+rw /dev/ttyUSB0`

| Component      | Length [cm] |
| -----------    | ----------- |
| Pen            | 14          |


## Robot answers examples:

- `LISTPV cur\r\nPosition CUR\r\n 1:0        2:-3923    3:0        4: 1       5:0       \r\n X: 6508    Y:-353     Z: 8278    P: 231     R:-201    \r\n>`

- `SPEED 5\r\n\x00Done.\r\n>`

## Extras

- `time` in command `MOVES points 1 len(points) time`, is `number_of_points * time_per_point` and is in ms