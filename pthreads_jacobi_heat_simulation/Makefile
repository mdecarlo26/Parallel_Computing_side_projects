CC = gcc
TARGET = solver
CFLAGS = -O3 -Wall -std=gnu99 -lpthread -lm
CHECK_TARGET = solver_gold
NUM_THREADS = 8
SIZE_ARR = 32
MIN = 100
MAX = 150

all: $(TARGET)
$(TARGET): $(TARGET).c
	$(CC) -o $(TARGET) $(TARGET).c  $(CHECK_TARGET).c $(CFLAGS)

run:$(TARGET)
	./$(TARGET) $(SIZE_ARR) $(NUM_THREADS) $(MIN) $(MAX)

debug: $(TARGET)
	valgrind --leak-check=yes -s ./$(TARGET) $(SIZE_ARR) $(NUM_THREADS)

clean:
	rm -f $(TARGET)

redo:
	clear
	rm -f $(TARGET)
	$(CC) -o $(TARGET) $(TARGET).c  $(CHECK_TARGET).c $(CFLAGS)
	./$(TARGET) $(SIZE_ARR) $(NUM_THREADS) $(MIN) $(MAX)
time:
	rm -f $(TARGET)
	$(CC) -o $(TARGET) $(TARGET).c  $(CFLAGS)
	time ./$(TARGET) $(SIZE_ARR) $(NUM_THREADS) $(MIN) $(MAX)

.PHONY: all clean debug run redo time 