# Pkg install avahi libopus

cmake_minimum_required(VERSION 3.1)
project(moonlight-embedded VERSION 2.5.3 LANGUAGES C)
SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
SET(CMAKE_C_STANDARD 99)
include(${CMAKE_ROOT}/Modules/GNUInstallDirs.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/generate_version_header.cmake)

aux_source_directory(./src SRC_LIST)
list(APPEND SRC_LIST ./src/input/evdev.c ./src/input/mapping.c ./src/input/udev.c)

set(MOONLIGHT_DEFINITIONS)

find_package(Opus REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(AVCODEC libavcodec)
pkg_check_modules(AVCODEC libswscale)
pkg_check_modules(AVUTIL libavutil)

list(APPEND MOONLIGHT_DEFINITIONS HAS_SOCKLEN_T)

SET(MOONLIGHT_COMMON_INCLUDE_DIR ./third_party/moonlight-common-c/src)
SET(GAMESTREAM_INCLUDE_DIR ./libgamestream)

include_directories("${PROJECT_BINARY_DIR}")
add_subdirectory(libgamestream)

add_executable(moonlight ${SRC_LIST})
target_link_libraries(moonlight m)
target_link_libraries(moonlight gamestream)

target_link_libraries(moonlight avcodec)
target_link_libraries(moonlight swscale)
target_include_directories(moonlight PRIVATE ${AVCODEC_INCLUDE_DIRS} ${AVUTIL_INCLUDE_DIRS})
target_link_libraries(moonlight ${AVCODEC_LIBRARIES} ${AVUTIL_LIBRARIES})

set_property(TARGET moonlight PROPERTY COMPILE_DEFINITIONS ${MOONLIGHT_DEFINITIONS})
target_include_directories(moonlight PRIVATE ${GAMESTREAM_INCLUDE_DIR} ${MOONLIGHT_COMMON_INCLUDE_DIR} ${OPUS_INCLUDE_DIRS})
target_link_libraries(moonlight ${OPUS_LIBRARY} ${CMAKE_DL_LIBS})

add_subdirectory(docs)

install(TARGETS moonlight DESTINATION ${CMAKE_INSTALL_BINDIR})
# install(FILES ./third_party/SDL_GameControllerDB/gamecontrollerdb.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/moonlight)
# install(FILES moonlight.conf DESTINATION ${CMAKE_INSTALL_SYSCONFDIR})
