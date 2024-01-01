#include "common.h"

#include <SPlisHSPlasH/TimeStep.h>
#include <SPlisHSPlasH/DiffDFSPH/SimulationDataDiffDFSPH.h>
#include <SPlisHSPlasH/DiffDFSPH/TimeStepDiffDFSPH.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

namespace py = pybind11;

void DiffDFSPHModule(py::module m_sub) {
    // ---------------------------------------
    // Class Simulation Data DiffDFSPH
    // ---------------------------------------
    py::class_<SPH::SimulationDataDiffDFSPH>(m_sub, "SimulationDataDiffDFSPH")
            .def(py::init<>())
            .def("init", &SPH::SimulationDataDiffDFSPH::init)
            .def("cleanup", &SPH::SimulationDataDiffDFSPH::cleanup)
            .def("reset", &SPH::SimulationDataDiffDFSPH::reset)
            .def("performNeighborhoodSearchSort", &SPH::SimulationDataDiffDFSPH::performNeighborhoodSearchSort)

            .def("getFactor", (const Real (SPH::SimulationDataDiffDFSPH::*)(const unsigned int, const unsigned int)const)(&SPH::SimulationDataDiffDFSPH::getFactor))
            // .def("getFactor", (Real& (SPH::SimulationDataDiffDFSPH::*)(const unsigned int, const unsigned int))(&SPH::SimulationDataDiffDFSPH::getFactor)) // TODO: wont work by reference
            .def("setFactor", &SPH::SimulationDataDiffDFSPH::setFactor)

            .def("getKappa", (const Real (SPH::SimulationDataDiffDFSPH::*)(const unsigned int, const unsigned int)const)(&SPH::SimulationDataDiffDFSPH::getKappa))
            // .def("getKappa", (Real& (SPH::SimulationDataDiffDFSPH::*)(const unsigned int, const unsigned int))(&SPH::SimulationDataDiffDFSPH::getKappa)) // TODO: wont work by reference
            .def("setKappa", &SPH::SimulationDataDiffDFSPH::setKappa)

            .def("getKappaV", (const Real (SPH::SimulationDataDiffDFSPH::*)(const unsigned int, const unsigned int)const)(&SPH::SimulationDataDiffDFSPH::getKappaV))
            // .def("getKappaV", (Real& (SPH::SimulationDataDiffDFSPH::*)(const unsigned int, const unsigned int))(&SPH::SimulationDataDiffDFSPH::getKappaV)) // TODO: wont work by reference
            .def("setKappaV", &SPH::SimulationDataDiffDFSPH::setKappaV)

            .def("getDensityAdv", (const Real (SPH::SimulationDataDiffDFSPH::*)(const unsigned int, const unsigned int)const)(&SPH::SimulationDataDiffDFSPH::getDensityAdv))
            // .def("getDensityAdv", (Real& (SPH::SimulationDataDiffDFSPH::*)(const unsigned int, const unsigned int))(&SPH::SimulationDataDiffDFSPH::getDensityAdv)) // TODO: wont work by reference
            .def("setDensityAdv", &SPH::SimulationDataDiffDFSPH::setDensityAdv);

    // ---------------------------------------
    // Class Time Step DiffDFSPH
    // ---------------------------------------
    py::class_<SPH::TimeStepDiffDFSPH, SPH::TimeStep>(m_sub, "TimeStepDiffDFSPH")
            .def_readwrite_static("SOLVER_ITERATIONS_V", &SPH::TimeStepDiffDFSPH::SOLVER_ITERATIONS_V)
            .def_readwrite_static("MAX_ITERATIONS_V", &SPH::TimeStepDiffDFSPH::MAX_ITERATIONS_V)
            .def_readwrite_static("MAX_ERROR_V", &SPH::TimeStepDiffDFSPH::MAX_ERROR_V)
            .def_readwrite_static("USE_DIVERGENCE_SOLVER", &SPH::TimeStepDiffDFSPH::USE_DIVERGENCE_SOLVER)

            .def_readwrite_static("TARGET_X", &SPH::TimeStepDiffDFSPH::TARGET_X)
            .def_readwrite_static("TARGET_TIME", &SPH::TimeStepDiffDFSPH::TARGET_TIME)
            .def_readwrite_static("INIT_V_RB", &SPH::TimeStepDiffDFSPH::INIT_V_RB)
           
            .def("get_boundary_model", &SPH::TimeStepDiffDFSPH::get_boundary_model_Akinci12, py::return_value_policy::reference_internal)

            .def("get_loss", &SPH::TimeStepDiffDFSPH::get_loss)
            .def("set_loss", &SPH::TimeStepDiffDFSPH::set_loss)
            .def("get_loss_x", &SPH::TimeStepDiffDFSPH::get_loss_x)
            .def("set_loss_x", &SPH::TimeStepDiffDFSPH::set_loss_x)
            .def("get_loss_rotation", &SPH::TimeStepDiffDFSPH::get_loss_rotation)
            .def("set_loss_rotation", &SPH::TimeStepDiffDFSPH::set_loss_rotation)
            .def("get_lr", &SPH::TimeStepDiffDFSPH::get_lr)
            .def("set_lr", &SPH::TimeStepDiffDFSPH::set_lr)
            
            .def("set_init_v_rb", &SPH::TimeStepDiffDFSPH::set_init_v_rb)
            .def("set_init_omega_rb", &SPH::TimeStepDiffDFSPH::set_init_omega_rb)
            .def("set_init_omega_rb_to_joint", &SPH::TimeStepDiffDFSPH::set_init_omega_rb_to_joint)

            .def("get_init_v_rb", &SPH::TimeStepDiffDFSPH::get_init_v_rb)
            .def("get_init_omega_rb", &SPH::TimeStepDiffDFSPH::get_init_omega_rb)

            .def("get_target_x", &SPH::TimeStepDiffDFSPH::get_target_x)
            .def("set_target_x", &SPH::TimeStepDiffDFSPH::set_target_x)
            .def("get_target_angle_in_radian", &SPH::TimeStepDiffDFSPH::get_target_angle_in_radian)
            .def("get_target_quaternion_vec4", &SPH::TimeStepDiffDFSPH::get_target_quaternion_vec4)
            
            .def("is_trajectory_finish_callback", &SPH::TimeStepDiffDFSPH::is_trajectory_finish_callback)
            .def("clear_all_callbacks", &SPH::TimeStepDiffDFSPH::clear_all_callbacks)
            .def("is_in_new_trajectory", &SPH::TimeStepDiffDFSPH::is_in_new_trajectory)

            .def("set_custom_log_message", &SPH::TimeStepDiffDFSPH::set_custom_log_message)
            .def("get_custom_log_message", &SPH::TimeStepDiffDFSPH::get_custom_log_message)

            .def("get_step_count", &SPH::TimeStepDiffDFSPH::get_step_count)
            .def("add_log", &SPH::TimeStepDiffDFSPH::add_log)
            
            .def("reset_gradient", &SPH::TimeStepDiffDFSPH::reset_gradient)

            .def("get_num_1ring_fluid_particle", &SPH::TimeStepDiffDFSPH::get_num_1ring_fluid_particle)

             // -------------------------------------------------------------------
            .def("setlbfgsMaxIter", &SPH::TimeStepDiffDFSPH::setlbfgsMaxIter)
            .def("setlbfgsMemorySize", &SPH::TimeStepDiffDFSPH::setlbfgsMemorySize)
            .def("setlbfgsLineSearchMethod", &SPH::TimeStepDiffDFSPH::setlbfgsLineSearchMethod)
            .def("setlbfgsMaxLineSearch", &SPH::TimeStepDiffDFSPH::setlbfgsMaxLineSearch)
            
            .def("setlbfgsFunction", &SPH::TimeStepDiffDFSPH::setlbfgsFunction)
            .def("setlbfgsInitPoint", &SPH::TimeStepDiffDFSPH::setlbfgsInitPoint)
            .def("startlbfgsTraining", &SPH::TimeStepDiffDFSPH::startlbfgsTraining)
            .def("setlbfgsUseNormalizedGrad", &SPH::TimeStepDiffDFSPH::setlbfgsUseNormalizedGrad)

            .def(py::init<>());
}
