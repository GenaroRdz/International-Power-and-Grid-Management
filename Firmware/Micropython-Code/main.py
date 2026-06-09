import sys
import asyncio
import setup
import functions
import INA226

# How often to push a fresh set of INA226 readings to the host, in seconds.
# 0.2 s = 5 readings per second. Raise it to stream slower, lower it for faster.
INA_PERIOD = 0.3

# How often to poll the CAN bus for incoming frames, in seconds. 0.05 s = 20x/s.
# This only LISTENS for frames addressed to / seen by this board and forwards
# them to the host as 'CANRX,...' lines. It is optional: if you only care about
# the commands MAIN sends out, you can drop can_rx_task() from the gather below.
CAN_RX_PERIOD = 0.05


async def command_task():
    sreader = asyncio.StreamReader(sys.stdin)
    while True:
        try:
            line = await sreader.readline()
            cmd = line.strip().decode()
            functions.handle_ping(cmd)
            functions.commands(cmd)
        except Exception:
            print("Commando no encontrado")


async def ina_task():
    while True:
        if functions.initialized and functions.streaming:
            try:
                print(INA226.read_all_INAs_str())
            except Exception as e:
                print("ERROR: INA read failed: {}".format(e))
        await asyncio.sleep(INA_PERIOD)


async def can_rx_task():
    # Drain any CAN frames this board received and forward them to the host as
    # 'CANRX,...' lines. Runs cooperatively with command_task on the same event
    # loop, so it never fights can_send for the SPI bus and never blocks pings.
    while True:
        try:
            functions.can_poll_rx()
        except Exception:
            pass
        await asyncio.sleep(CAN_RX_PERIOD)


async def main():
    # Reset all outputs to their default state before starting
    functions.reset()
    print("Sistema listo")
    print("BOOT")

    # Run the command listener, the INA streamer and the CAN receiver side by
    # side. To disable CAN receive, simply remove can_rx_task() from this call.
    await asyncio.gather(command_task(), ina_task(), can_rx_task())


asyncio.run(main())
