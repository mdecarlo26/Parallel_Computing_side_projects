CC = gcc
TARGET = gauss_eliminate
CFLAGS = -O3 -Wall -std=gnu99 -lpthread -lm
CHECK_TARGET = compute_gold
NUM_THREADS = 4
SIZE_ARR = 512


all: $(TARGET)
$(TARGET): $(TARGET).c
	$(CC) -o $(TARGET) $(TARGET).c  $(CHECK_TARGET).c $(CFLAGS)

run:$(TARGET)
	./$(TARGET) $(SIZE_ARR) $(NUM_THREADS)

debug: $(TARGET)
	valgrind --leak-check=yes -s ./$(TARGET) $(SIZE_ARR) $(NUM_THREADS)

clean:
	rm -f $(TARGET)

redo:
	clear
	rm -f $(TARGET)
	$(CC) -o $(TARGET) $(TARGET).c  $(CHECK_TARGET).c $(CFLAGS)
	./$(TARGET) $(SIZE_ARR) $(NUM_THREADS) 2>dummy.txt
time:
	rm -f $(TARGET)
	$(CC) -o $(TARGET) $(TARGET).c  $(CFLAGS)
	time ./$(TARGET) $(SIZE_ARR) $(NUM_THREADS)

.PHONY: all clean debug run redo time 