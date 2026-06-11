from can_setup import can, Current_Device, ADDR, ADDR_NAMES, id_of, name_of


def receive_message():
    if can.check_receive():
        msg_id, data = can.read_message()
        if len(data) >= 1:
            sender = name_of(msg_id)                # who sent it (from the ID)
            target = ADDR_NAMES.get(data[0], "?")   # who it's for (byte 0)
            text   = bytes(data[1:]).decode()       # the command text
            return sender, target, text
    return None


def send_message(target, text):
    payload = bytes([ADDR[target]]) + text.encode()
    try:
        can.send_message(id_of(Current_Device), payload)
        return True
    except Exception:
        return False 


