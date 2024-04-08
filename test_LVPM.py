import usb.core

def device_matcher(d):
    try:
        return d.idVendor == 0x2AB9 and d.idProduct == 0x0001 and (serialno is None or d.serial_number == str(serialno))
    except:#Catches some platform-specific errors when connecting to multiple PMs simultaneously.
        return False
DEVICE = usb.core.find(custom_match=device_matcher)

print(DEVICE)