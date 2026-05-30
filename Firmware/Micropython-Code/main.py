import sys
import asyncio
import setup
import functions
import INA226

# How often to push a fresh set of INA226 readings to the host, in seconds.
# 0.2 s = 5 readings per second. Raise it to stream slower, lower it for faster.
INA_PERIOD = 0.3


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


async def main():
    # Reset all outputs to their default state before starting
    functions.reset()
    print("Sistema listo")
    print("BOOT")

    # Run the command listener and the INA streamer side by side.
    await asyncio.gather(command_task(), ina_task())


asyncio.run(main())
