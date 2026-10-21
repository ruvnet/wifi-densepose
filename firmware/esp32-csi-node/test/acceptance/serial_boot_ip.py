#!/usr/bin/env python3
"""Read an ESP-IDF boot log from a serial port and report the node IP.

Standard library only (termios / fcntl / os), so it runs on any POSIX host
without pyserial installed.

Prints exactly one of these on stdout:

    IP=192.168.0.115     the node came up and got this address
    DIED                  the boot log shows ESP_ERROR_CHECK failed
    TIMEOUT               no IP appeared before the timeout

The exit code is 0 for IP, 3 for DIED and 4 for TIMEOUT, so a shell script
can branch on it without parsing the text.

    ./serial_boot_ip.py --port /dev/ttyUSB0 --timeout 30
    ./serial_boot_ip.py --port /dev/ttyUSB0 --no-reset   # just read

On a pty (for testing) the same code path runs; --no-reset is the useful
mode there, because a pty has no modem control lines to reset the board
with.
"""

import argparse
import fcntl
import os
import re
import select
import sys
import termios
import time

IP_RE = re.compile(rb"Got IP: (\d+\.\d+\.\d+\.\d+)")
DEAD_RE = re.compile(rb"ESP_ERROR_CHECK failed")

EXIT_IP = 0
EXIT_DEAD = 3
EXIT_TIMEOUT = 4


def configure(fd, baud):
    """115200 8N1, raw, no echo, no flow control."""
    iflag, oflag, cflag, lflag, ispeed, ospeed, cc = termios.tcgetattr(fd)
    iflag &= ~(termios.INPCK | termios.ISTRIP | termios.IXON | termios.IXOFF
               | termios.ICRNL | termios.INLCR)
    oflag &= ~termios.OPOST
    lflag &= ~(termios.ECHO | termios.ICANON | termios.ISIG | termios.IEXTEN)
    cflag &= ~(termios.CSIZE | termios.PARENB | termios.CSTOPB)
    cflag |= termios.CS8 | termios.CREAD | termios.CLOCAL
    speed = getattr(termios, "B%d" % baud, None)
    if speed is None:
        raise SystemExit("unsupported baud: %d" % baud)
    cc[termios.VMIN] = 0
    cc[termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW,
                      [iflag, oflag, cflag, lflag, speed, speed, cc])
    termios.tcflush(fd, termios.TCIFLUSH)


def get_modem(fd):
    return bytearray(fcntl.ioctl(fd, termios.TIOCMGET, b"\x00" * 4))


def set_modem(fd, bits):
    fcntl.ioctl(fd, termios.TIOCMSET, bytes(bits))


def pulse_reset(fd, pulse=0.12):
    """Reset the board the way the .NET version did: DTR stays low and RTS
    is pulsed high. On an ESP32 DevKit the RTS line drives EN.

    Returns False when the port has no modem control lines (a pty, or a
    serial adapter without them) - there is nothing to reset in that case.
    """
    try:
        bits = get_modem(fd)
    except OSError:
        return False
    try:
        bits[0] &= ~termios.TIOCM_DTR
        bits[0] &= ~termios.TIOCM_RTS
        set_modem(fd, bits)
        time.sleep(0.3)
        bits[0] |= termios.TIOCM_RTS
        set_modem(fd, bits)
        time.sleep(pulse)
        bits[0] &= ~termios.TIOCM_RTS
        set_modem(fd, bits)
    except OSError:
        return False
    return True


def read_boot(fd, timeout):
    """Read until the IP shows up, the node dies, or the timeout expires."""
    deadline = time.time() + timeout
    buf = b""
    while time.time() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.1)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 4096)
        except BlockingIOError:
            # the fd is non-blocking and select() said it was ready; a
            # spurious wakeup is not an error
            continue
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
        found = IP_RE.search(buf)
        if found:
            return "IP", found.group(1).decode("ascii"), buf
        if DEAD_RE.search(buf):
            return "DIED", None, buf
    return "TIMEOUT", None, buf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", required=True, help="serial device, e.g. /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="seconds to wait for the boot log")
    ap.add_argument("--no-reset", action="store_true",
                    help="do not pulse RTS before reading")
    ap.add_argument("--log", help="write the raw boot log to this file")
    args = ap.parse_args()

    fd = os.open(args.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        configure(fd, args.baud)
        if not args.no_reset:
            pulse_reset(fd)
        status, ip, buf = read_boot(fd, args.timeout)
    finally:
        os.close(fd)

    if args.log:
        with open(args.log, "wb") as fh:
            fh.write(buf)

    if status == "IP":
        print("IP=%s" % ip)
        return EXIT_IP
    if status == "DIED":
        print("DIED")
        return EXIT_DEAD
    print("TIMEOUT")
    return EXIT_TIMEOUT


if __name__ == "__main__":
    sys.exit(main())
