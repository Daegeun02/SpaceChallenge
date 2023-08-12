from threading import Thread

from CrazyDG.communication_test import CommunicationCenter

from time import sleep



class CommunicationCenter_RE( CommunicationCenter ):

    def run( self ):

        packet = self.packet
        rxData = packet.RxData

        dt = 1 / self.Hz

        print( 'communication start' )

        if self.packet is not None:

            packet = self.packet
            packet._enroll( 3, self.Txheader )

        print( 'tx started' )

        txData = packet.TxData

        while self.AllGreen:

            packet._sendto()

            print( txData )

            sleep( dt )