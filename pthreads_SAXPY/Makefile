CC = gcc
TARGET = saxpy
CFLAGS = -O3 -Wall -std=c99 -lpthread -lm
NUM_THREADS = 8
SIZE_ARR = 100000000

all: $(TARGET)
$(TARGET): $(TARGET).c
	$(CC) -o $(TARGET) $(TARGET).c  $(CFLAGS)

run:$(TARGET)
	./$(TARGET) $(SIZE_ARR) $(NUM_THREADS)

debug: $(TARGET)
	valgrind --leak-check=yes -s ./$(TARGET) $(SIZE_ARR) $(NUM_THREADS)

clean:
	rm -f $(TARGET)

redo:
	clear
	rm -f $(TARGET)
	$(CC) -o $(TARGET) $(TARGET).c  $(CFLAGS)
	./$(TARGET) $(SIZE_ARR) $(NUM_THREADS)
time:
	rm -f $(TARGET)
	$(CC) -o $(TARGET) $(TARGET).c  $(CFLAGS)
	time ./$(TARGET) $(SIZE_ARR) $(NUM_THREADS)

.PHONY: all clean debug run redo time