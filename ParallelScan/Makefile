NVCC=nvcc 

OPENCV_INCLUDEPATH=/usr/local/include/opencv4
OPENCV_LIBPATH=/usr/local/lib 

OPENCV_LIBS=-L -lopencv_dnn -lopencv_ml -lopencv_objdetect -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_videostab -lopencv_video -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_flann -lopencv_core
OPENCV_CFLAGS=`pkg-config --cflags opencv4`

CUDA_INCLUDEPATH=/usr/local/cuda/include

NVCC_OPTS=-arch=sm_30 
GCC_OPTS=-std=c++11 -g -O0 -Wall 
CUDA_LD_FLAGS=-L -lcuda -lcudart

final: main.o scan.o
	g++ -o scan main.o scan.o $(CUDA_LD_FLAGS) $(OPENCV_LIBS)

main.o: main.cpp scan.h utils.h 
	g++ -c $(GCC_OPTS) main.cpp  $(OPENCV_LIBS) 

scan.o: scan.cu scan.h utils.h
	$(NVCC) -c scan.cu $(NVCC_OPTS)

clean:
	rm *.o scan
