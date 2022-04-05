SSEFLAGS = -O3 -mavx2 -fopenmp -Wno-misleading-indentation 
BUILD_DIRECTORY = build/

FILE = mandelbrot_sse_avx
ADDITIONAL_FILES = ../src/TXlib.h

execute: $(FILE).exe
	./$(BUILD_DIRECTORY)$(FILE).exe

$(FILE).exe: $(FILE).o
	g++ $(BUILD_DIRECTORY)$(FILE).o -o $(BUILD_DIRECTORY)$(FILE).exe $(SSEFLAGS)

$(FILE).o: $(FILE).cpp $(ADDITIONAL_FILES)
	g++ -c $(FILE).cpp -o $(BUILD_DIRECTORY)$(FILE).o $(SSEFLAGS) 