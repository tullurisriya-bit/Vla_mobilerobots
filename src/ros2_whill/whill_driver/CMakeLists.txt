cmake_minimum_required(VERSION 3.8)
project(whill_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(whill_msgs REQUIRED)

include_directories(include
  ${whill_msgs_INCLUDE_DIRS}
)

add_executable(whill
  src/entrypoint.cpp
  src/whill_node.cpp
  src/model_cr2/whill.cpp
  src/model_cr2/parser.cpp
  src/model_cr2/hardware/serial_port.cpp
)
ament_target_dependencies(whill
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  geometry_msgs
  whill_msgs
)

add_library(whill_node SHARED
  src/whill_node.cpp
  src/model_cr2/whill.cpp
  src/model_cr2/parser.cpp
  src/model_cr2/hardware/serial_port.cpp
)
target_compile_definitions(whill_node
  PRIVATE "COMPOSITION_BUILDING_DLL"
)
ament_target_dependencies(whill_node
  rclcpp
  rclcpp_components
  std_msgs
  sensor_msgs
  geometry_msgs
  "whill_msgs"
)
rclcpp_components_register_nodes(whill_node "whill_driver::WhillNode")

install(TARGETS
  whill
  whill_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
