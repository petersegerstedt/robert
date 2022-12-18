def loads(s):
    lines = [l.strip() for l in s.split('\n') if len(l.strip())]
    if not lines:
        raise Exception('no data')
    if not 'JASC-PAL' == lines[0]:
        raise Exception('not JASC-PAL')
    if 3 > len(lines):
        raise Exception('invalid')
    n = int(lines[2])
    if not n == len(lines[3:]):
        raise Exception(f'invalid, expected {n} entries, found {len(lines[3:])}')
    return [tuple(int(t) for t in l.split(' ')) for l in lines[3:]]
def load(f):
    return loads(f.read())
    
if '__main__' == __name__:
    import sys
    with open(sys.argv[1], 'r') as f:
        print(load(f))