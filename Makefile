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

######################################################################
# Define current working source directory (added)
currentDirectory = src/V1

######################################################################
# There should be no need to modify anything below this line (but
# feel free to if you want).

EXAMPLES = $(currentDirectory)/example1.c $(currentDirectory)/example2.c $(currentDirectory)/example3.c \
           $(currentDirectory)/example4.c $(currentDirectory)/example5.c
ARCH = $(currentDirectory)/convolve.c $(currentDirectory)/error.c $(currentDirectory)/pnmio.c \
       $(currentDirectory)/pyramid.c $(currentDirectory)/selectGoodFeatures.c \
       $(currentDirectory)/storeFeatures.c $(currentDirectory)/trackFeatures.c \
       $(currentDirectory)/klt.c $(currentDirectory)/klt_util.c $(currentDirectory)/writeFeatures.c
LIB = -L/usr/local/lib -L/usr/lib

.SUFFIXES:  .c .o

all:  lib $(notdir $(EXAMPLES:.c=))

$(currentDirectory)/%.o: $(currentDirectory)/%.c
	$(CC) -c $(CFLAGS) $< -o $@

lib: $(ARCH:.c=.o)
	rm -f $(currentDirectory)/libklt.a
	ar ruv $(currentDirectory)/libklt.a $(ARCH:.c=.o)
	rm -f $(currentDirectory)/*.o

example1: $(currentDirectory)/libklt.a
	$(CC) -O3 $(CFLAGS) -o $(currentDirectory)/example1 $(currentDirectory)/example1.c -L$(currentDirectory) -lklt $(LIB) -lm

example2: $(currentDirectory)/libklt.a
	$(CC) -O3 $(CFLAGS) -o $(currentDirectory)/example2 $(currentDirectory)/example2.c -L$(currentDirectory) -lklt $(LIB) -lm

example3: $(currentDirectory)/libklt.a
	$(CC) -O3 $(CFLAGS) -o $(currentDirectory)/example3 $(currentDirectory)/example3.c -L$(currentDirectory) -lklt $(LIB) -lm

example4: $(currentDirectory)/libklt.a
	$(CC) -O3 $(CFLAGS) -o $(currentDirectory)/example4 $(currentDirectory)/example4.c -L$(currentDirectory) -lklt $(LIB) -lm

example5: $(currentDirectory)/libklt.a
	$(CC) -O3 $(CFLAGS) -o $(currentDirectory)/example5 $(currentDirectory)/example5.c -L$(currentDirectory) -lklt $(LIB) -lm

depend:
	makedepend $(ARCH) $(EXAMPLES)

clean:
	rm -f $(currentDirectory)/*.o $(currentDirectory)/*.a $(currentDirectory)/example* \
	      *.tar *.tar.gz $(currentDirectory)/libklt.a feat*.ppm features.ft features.txt

