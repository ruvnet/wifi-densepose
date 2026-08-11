# SPDX-License-Identifier: MIT OR Apache-2.0
# Host build/test for the portable veil_shield core (no hardware).
CC ?= cc
CFLAGS ?= -std=c99 -Wall -Wextra -Werror -O2
LDLIBS ?= -lm

.PHONY: test clean
test: test_veil_shield
	./test_veil_shield

test_veil_shield: test/test_veil_shield.c veil_shield.c veil_shield.h
	$(CC) $(CFLAGS) -o $@ test/test_veil_shield.c veil_shield.c $(LDLIBS)

clean:
	rm -f test_veil_shield
