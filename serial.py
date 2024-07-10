# serial bluetooth with hc-06

import serial

robot = serial.Serial("COM1", 9600, timeout =1)

def getstate ():
    robot.write("state") #when robot get bit 1, return sensor value
    data = robot.readline().decode('ascii')
    return data

def action(h1, v1, h2, v2): # four angles, horizontal & vertical
    robot.write(%)
    if robot.readline().decode('ascii')=="error":
        robot.write("reset")
        end episode #end episode
    else:
        pass



