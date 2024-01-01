if(BUILD_SHARED_LIBS)
	set(LIB_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
	set(LIB_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
else(BUILD_SHARED_LIBS)
	set(LIB_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
	set(LIB_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
endif(BUILD_SHARED_LIBS)

## NeighborhoodSearch
include(NeighborhoodSearch)

## Discregrid
ExternalProject_Add(
	Ext_Discregrid
	PREFIX "${CMAKE_BINARY_DIR}/extern/Discregrid"
	GIT_REPOSITORY https://github.com/InteractiveComputerGraphics/Discregrid.git
	GIT_TAG "0b69062ff9c56fbb6dcecd296652028bedbacf0e"
	INSTALL_DIR ${ExternalInstallDir}/Discregrid
	CMAKE_ARGS -DCMAKE_BUILD_TYPE:STRING=${EXT_CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX:PATH=${ExternalInstallDir}/Discregrid -DBUILD_CMD_EXECUTABLE:BOOL=0 -DEIGEN3_INCLUDE_DIR:PATH=${EIGEN3_INCLUDE_DIR} -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
)
ExternalProject_Get_Property(Ext_Discregrid INSTALL_DIR)
set(DISCREGRID_INCLUDE_DIR ${INSTALL_DIR}/include)
set(DISCREGRID_DEBUG_LIB ${INSTALL_DIR}/lib/${LIB_PREFIX}Discregrid_d${LIB_SUFFIX})
set(DISCREGRID_LIB ${INSTALL_DIR}/lib/${LIB_PREFIX}Discregrid${LIB_SUFFIX})
set(DISCREGRID_LIBRARIES
	optimized ${DISCREGRID_LIB}
	debug ${DISCREGRID_DEBUG_LIB}
)
unset(INSTALL_DIR)

## GenericParameters
ExternalProject_Add(
	Ext_GenericParameters
	PREFIX "${CMAKE_BINARY_DIR}/extern/GenericParameters"
	GIT_REPOSITORY https://github.com/InteractiveComputerGraphics/GenericParameters.git
	GIT_TAG "42d52ad551fafba600ee99e59fb0f9c7b557e2ed"
	INSTALL_DIR ${ExternalInstallDir}/GenericParameters
	CMAKE_ARGS -DCMAKE_BUILD_TYPE=${EXT_CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX:PATH=${ExternalInstallDir}/GenericParameters -DGENERICPARAMETERS_NO_TESTS:BOOL=1 -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
)
ExternalProject_Get_Property(Ext_GenericParameters INSTALL_DIR)
set(GENERICPARAMETERS_INCLUDE_DIR ${INSTALL_DIR}/include)
unset(INSTALL_DIR)

