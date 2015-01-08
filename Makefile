.SUFFIXES: .c .u
CC= gcc
CFLAGS= -O3 -Wall -g
LDFLAGS= -lm

LOBJECTS= main.o PTM-model.o cokus.o

LSOURCE= main.c PTM-model.c cokus.c

ptm:	$(LOBJECTS)
	$(CC) $(CFLAGS) $(LOBJECTS) -o ptm $(LDFLAGS)

clean:
	-rm -f *.o
