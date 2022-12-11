import signal
import time

[signal.SIGINT, signal.SIGILL, signal.SIGFPE, signal.SIGSEGV,
    signal.SIGTERM, signal.SIGBREAK, signal.SIGABRT]

# Our signal handler


def exit_handler(signum, frame):
    print("Signal Number:", signum, " Frame: ", frame)
    exit(0)


# Register our signal handler with `SIGINT`(CTRL + C)
signal.signal(signal.SIGINT, exit_handler)

# While Loop
while 1:
    print("Press Ctrl + C")
    time.sleep(3)
