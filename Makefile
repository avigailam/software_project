.PHONY: all clean

CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

SOURCES = symnmf.c
HEADERS = symnmf.h
OBJECTS = $(SOURCES:.c=.o)
TARGET = symnmf

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) -lm

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)
