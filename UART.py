import serial

from threading import Thread

import time


port = "/dev/ttyTHS0"

baud = 115200

ser = serial.Serial( port, baud, timeout=1 )


def main():

    thread = Thread( target=readthread, args=(ser,), daemon=True )

    thread.start()

    while True:

        data = "str.decode()".encode()

        # ser.write( data )

        time.sleep( 1 )


def readthread( ser ):

    while True:

        if ser.readable():

            res = ser.readline()

            print( res )

    ser.close()


if __name__ == "__main__":

    main()
