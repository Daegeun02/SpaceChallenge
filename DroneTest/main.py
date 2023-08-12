from CrazyDG import CrazyDragon

# from CrazyDG.communication_test import CommunicationCenter
from correction import CommunicationCenter_RE

from numpy import array

from time import sleep



Txheader = array([26, 4])
Rxheader = array([ 4,26])


config = {
    'Hz'      : 10,
    'Txheader': Txheader,
    'Rxheader': Rxheader,
    'port'    : '/dev/ttyTHS0',
    'baud'    : 115200
}


def Guidance_TEST( _cf, config ):

    CMC = CommunicationCenter_RE( _cf, config )
    CMC.start()

    input( 'exit' )

    CMC.AllGreen = False

    CMC.join()



if __name__ == "__main__":

    _cf = CrazyDragon()

    Guidance_TEST( _cf, config )