TARGET_EXEC ?= cuda-flou

BUILD_DIR ?= ./build
SRC_DIRS ?= ./src

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.cu -or -name *.c -or -name *.s)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CCU := nvcc
CUFLAGS ?= $(INC_FLAGS) -MMD -MP
CUFLAGS += --compiler-bindir /apps/2021/gcc/10.2/bin/
CUFLAGS += --std=c++17
LDFLAGS += -lcuda -lcudart

all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJS)
	$(CCU) $(CUFLAGS) $(OBJS) -o $@ $(LDFLAGS)

# c source
$(BUILD_DIR)/%.c.o: %.c
	@$(MKDIR_P) $(dir $@)
	$(CCU) $(CUFLAGS) $(NOCUDA) -c $< -o $@

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	@$(MKDIR_P) $(dir $@)
	$(CCU) $(CUFLAGS) $(NOCUDA) -c $< -o $@

# cuda source
$(BUILD_DIR)/%.cu.o: %.cu
	@$(MKDIR_P) $(dir $@)
	$(CCU) $(CUFLAGS) -c $< -o $@

.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)
	$(RM) $(TARGET_EXEC)	

-include $(DEPS)

MKDIR_P ?= mkdir -p