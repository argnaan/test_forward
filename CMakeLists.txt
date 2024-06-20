cmake_minimum_required(VERSION 3.19)

###############################################################################
# Panel Control
###############################################################################
set(TARGET_NAME "pulp_llama2")

#set(DATA_TYPE "fp32")
set(DATA_TYPE "fp16")

set(TRAIN_LIB "/home/andrea/pulp-trainlib/lib")
set(TRAIN_LIB_SRCS ${TRAIN_LIB}/sources)

set(TARGET_SRCS main.c)

if(DATA_TYPE STREQUAL "fp32")
	message(STATUS "Using fp32 version")
	list(APPEND TARGET_SRCS fp32/net.c)
	list(APPEND TARGET_SRCS fp32/llama2_utils.c)
	list(APPEND TARGET_SRCS ${TRAIN_LIB_SRCS}/pulp_matmul_fp32.c)
	list(APPEND TARGET_SRCS ${TRAIN_LIB_SRCS}/pulp_train_utils_fp32.c)
	list(APPEND TARGET_SRCS ${TRAIN_LIB_SRCS}/pulp_rmsnorm_fp32.c)
	list(APPEND TARGET_SRCS ${TRAIN_LIB_SRCS}/pulp_act_fp32.c)
else()
	message(STATUS "Using fp16 version")
	list(APPEND TARGET_SRCS fp16/net.c)
	list(APPEND TARGET_SRCS fp16/llama2_utils_fp16.c)
	list(APPEND TARGET_SRCS ${TRAIN_LIB_SRCS}/pulp_matmul_fp16.c)
	list(APPEND TARGET_SRCS ${TRAIN_LIB_SRCS}/pulp_train_utils_fp16.c)
	list(APPEND TARGET_SRCS ${TRAIN_LIB_SRCS}/pulp_rmsnorm_fp16.c)
	list(APPEND TARGET_SRCS ${TRAIN_LIB_SRCS}/pulp_act_fp16.c)
endif()

set(CONFIG_GAP_SDK_HOME $ENV{GAP_SDK_HOME})

set(CONFIG_DISABLE_WERROR TRUE)

set(GCC_LD_FLAGS "-lc")

include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)
	
set(INCLUDE_DIRS "${TRAIN_LIB}/include")

list(APPEND INCLUDE_DIRS "./")

###############################################################################
# CMake pre initialization
###############################################################################

include_directories(${INCLUDE_DIRS})

project(${TARGET_NAME} C ASM)
add_executable(${TARGET_NAME} ${TARGET_SRCS})

###############################################################################
# App's options interpretation
###############################################################################

target_compile_options(${TARGET_NAME} PRIVATE "-DNUM_CORES=8")
target_compile_options(${TARGET_NAME} PRIVATE "-DDATA_TYPE=${DATA_TYPE}")
target_compile_options(${TARGET_NAME} PRIVATE "-DFASTEXPF")
target_compile_options(${TARGET_NAME} PRIVATE "-DQ_RSQRT")

#target_compile_options(${TARGET_NAME} PRIVATE "-DOUTPUT")
#target_compile_options(${TARGET_NAME} PRIVATE "-DSTATS")
list(APPEND TARGET_CFLAGS -O3)

###############################################################################
# CMake post initialization
###############################################################################
setupos(${TARGET_NAME})
