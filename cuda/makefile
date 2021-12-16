######################
###### Variables #####
######################

#Compiler 
COMP=nvcc
#Compiler flags
FLGS=-g -G -arch=sm_60

#Directories
SRCDIR=src
OBJDIR=obj
TESTSDIR=tests

#Source files
SRCS=$(wildcard $(SRCDIR)/*.cu)
#Objects files
OBJS=$(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(SRCS))
#Tests files
TESTS=$(wildcard $(TESTSDIR)/*.cu)
TESTBINS=$(patsubst $(TESTSDIR)/%.cu,$(TESTSDIR)/bin/%,$(TESTS))

#Finall binary name
BINDIR=bin
BIN=$(BINDIR)/PCF

all: $(BIN)

release: FLGS=-O2 -arch=sm_60 -dlto
release: $(BIN)

$(BIN): $(OBJS)
	$(COMP) $(FLGS) $(OBJS) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(COMP) $(FLGS) -dc $< -o $@

#Create tests
$(TESTSDIR)/bin/%: $(TESTSDIR)/%.cu
	$(COMP) $(FLGS) $< $(OBJS) -o $@ -lcriterion

test: OBJS=obj/create_grid.o
test: $(OBJS) $(TESTBINS)
	for test in $(TESTBINS) ; do ./$$test ;  done

clean:
	rm -rf $(BINDIR)/* $(OBJDIR)/* $(TESTSDIR)/bin/* ../results/*.dat
