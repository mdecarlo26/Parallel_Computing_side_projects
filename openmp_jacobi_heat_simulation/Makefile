CC = gcc
TARGET = solver
CFLAGS = -O3 -Wall -std=c99 -lm -fopenmp
CHECK_TARGET = solver_gold
NUM_THREADS = 4
SIZE_ARR = 256
MIN = 1000
MAX = 1500

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
profile:
	clear
	rm -f $(TARGET)
	$(CC) -o $(TARGET) $(TARGET).c  $(CHECK_TARGET).c $(CFLAGS) -pg
	./$(TARGET) $(SIZE_ARR) $(NUM_THREADS) $(MIN) $(MAX)

analysis:
	gprof $(TARGET) gmon.out > analysis.txt

.PHONY: all clean debug run redo time profile analysis