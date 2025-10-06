######################################################################
# Choose your favorite C compiler
CC = gcc

######################################################################
# -DNDEBUG prevents the assert() statements from being included in 
# the code.  If you are having problems running the code, you might 
# want to comment this line to see if an assert() statement fires.
FLAG1 = -DNDEBUG

######################################################################
# -DKLT_USE_QSORT forces the code to use the standard qsort() 
# routine.  Otherwise it will use a quicksort routine that takes
# advantage of our specific data structure to greatly reduce the
# running time on some machines.  Uncomment this line if for some
# reason you are unhappy with the special routine.
# FLAG2 = -DKLT_USE_QSORT

######################################################################
# Add your favorite C flags here.
CFLAGS = $(FLAG1) $(FLAG2)
LDFLAGS = -pg   

######################################################################
# There should be no need to modify anything below this line (but
# feel free to if you want).

SRC_DIR = src/V1

EXAMPLES = $(SRC_DIR)/example1.c $(SRC_DIR)/example2.c $(SRC_DIR)/example3.c \
           $(SRC_DIR)/example4.c $(SRC_DIR)/example5.c

ARCH = $(SRC_DIR)/convolve.c $(SRC_DIR)/error.c $(SRC_DIR)/pnmio.c \
       $(SRC_DIR)/pyramid.c $(SRC_DIR)/selectGoodFeatures.c \
       $(SRC_DIR)/storeFeatures.c $(SRC_DIR)/trackFeatures.c \
       $(SRC_DIR)/klt.c $(SRC_DIR)/klt_util.c $(SRC_DIR)/writeFeatures.c

LIB = -L/usr/local/lib -L/usr/lib

.SUFFIXES:  .c .o

all: lib $(EXAMPLES:.c=)

# Compile .c to .o
$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) -c $(CFLAGS) -pg $< -o $@

# Build static library
lib: $(ARCH:.c=.o)
	rm -f $(SRC_DIR)/libklt.a
	ar ruv $(SRC_DIR)/libklt.a $(ARCH:.c=.o)
	rm -f $(SRC_DIR)/*.o

# Compile examples with profiler support
$(SRC_DIR)/example%: $(SRC_DIR)/libklt.a
	$(CC) -O3 $(CFLAGS) -pg -o $@ $@.c -L$(SRC_DIR) -lklt $(LIB) -lm $(LDFLAGS)


depend:
	makedepend $(ARCH) $(EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt profile.txt *.out
