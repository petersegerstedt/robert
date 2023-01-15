import struct
EVT_FMT  = 'llHHI'
EVT_SIZE = struct.calcsize(EVT_FMT)

EVT_TYPE_BTN = 1
BTNS = {272: 'left', 273: 'right', 274: 'middle'}


if __name__ == '__main__':
    import sys
    device = sys.argv[1]
    with open(device, 'rb') as f:
        while True:
            seconds, microseconds, type, code, value = struct.unpack(EVT_FMT, f.read(EVT_SIZE))
            if not EVT_TYPE_BTN == type: continue
            if not code in BTNS: continue
            if not value in [0, 1]: continue
            print(BTNS[code], 'button', ['up', 'down'][value]) 

