#include "TimeStepDiffDFSPH.h"
#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/FluidModel.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "SPlisHSPlasH/TimeStep.h"
#include "SPlisHSPlasH/Utilities/AVX_math.h"
#include "SimulationDataDiffDFSPH.h"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "Utilities/Timing.h"
#include "Utilities/Counting.h"
#include "Utilities/Logger.h"
#include "Utilities/ColorfulPrint.h"
#include "SPlisHSPlasH/Simulation.h"
#include "SPlisHSPlasH/BoundaryModel_Akinci2012.h"
#include "SPlisHSPlasH/BoundaryModel_Koschier2017.h"
#include "SPlisHSPlasH/BoundaryModel_Bender2019.h"
#include "SPlisHSPlasH/GradientUtils.h"

/* Loop over all dynamic boundary models
*/
#define forall_dynamic_boundary_model_Akinci12(code) \
for(int bm_index = 0; bm_index < sim->numberOfBoundaryModels(); bm_index++) \
  { \
    BoundaryModel_Akinci2012 *bm_i = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(bm_index)); \
    if(bm_i->getRigidBodyObject()->isDynamic())\
    {\
      code \
    }\
  }

#define forall_boundary_model_Akinci12(code) \
for(int bm_index = 0; bm_index < sim->numberOfBoundaryModels(); bm_index++) \
  { \
    BoundaryModel_Akinci2012 *bm_i = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(bm_index)); \
    code \
  }

using namespace SPH;
using namespace std;
using namespace GenParam;
using namespace Utilities;

#define STOP_TIMING_DEBUG STOP_TIMING_PRINT
// #define STOP_TIMING_DETAIL(name)                                     \
//   {                                                                  \
//     double time = STOP_TIMING;                                       \
//     NO_CONSOLE_LOG_INFO << "time " << name << ": " << time << " ms"; \
//   }
#define STOP_TIMING_DETAIL(name) STOP_TIMING

int TimeStepDiffDFSPH::SOLVER_ITERATIONS_V = -1;
int TimeStepDiffDFSPH::MAX_ITERATIONS_V = -1;
int TimeStepDiffDFSPH::MAX_ERROR_V = -1;
int TimeStepDiffDFSPH::USE_DIVERGENCE_SOLVER = -1;
int TimeStepDiffDFSPH::TARGET_X = -1;
int TimeStepDiffDFSPH::TARGET_ANGLE = -1;
int TimeStepDiffDFSPH::CURRENT_X_RB = -1;
int TimeStepDiffDFSPH::FINAL_X_RB = -1;
int TimeStepDiffDFSPH::INIT_V_RB = -1;
int TimeStepDiffDFSPH::FINAL_ANGLE_RB = -1;
int TimeStepDiffDFSPH::GRAD_INIT_V_RB = -1;
int TimeStepDiffDFSPH::INIT_OMEGA_RB = -1;
int TimeStepDiffDFSPH::GRAD_INIT_OMEGA_RB = -1;
int TimeStepDiffDFSPH::LOSS = -1;
int TimeStepDiffDFSPH::TARGET_TIME = -1;
int TimeStepDiffDFSPH::LR = -1;
int TimeStepDiffDFSPH::N_ITER = -1;
int TimeStepDiffDFSPH::DEBUG = -1;
int TimeStepDiffDFSPH::OPTIMIZE_ROTATION = -1;
int TimeStepDiffDFSPH::TARGET_QUATERNION = -1;
int TimeStepDiffDFSPH::FINAL_QUATERNION_RB = -1;
int TimeStepDiffDFSPH::CURRENT_OMEGA_RB = -1;
int TimeStepDiffDFSPH::CURRENT_V_RB = -1;
int TimeStepDiffDFSPH::TOTAL_MASS_RB = -1;
int TimeStepDiffDFSPH::TOTAL_MASS_FLUID = -1;
int TimeStepDiffDFSPH::UNIFORM_ACC_RB_TIME = -1;
int TimeStepDiffDFSPH::USE_MLS_PRESSURE = -1;
int TimeStepDiffDFSPH::USE_SYMMETRIC_FORMULA = -1;
int TimeStepDiffDFSPH::USE_STRONG_COUPLING = -1;
int TimeStepDiffDFSPH::USE_PRESSURE_WARMSTART = -1;
int TimeStepDiffDFSPH::USE_DIV_WARMSTART = -1;
int TimeStepDiffDFSPH::ENABLE_COUNT_NEIGHBOR_DOF = -1;

Vector3r total_force = Vector3r::Zero(); 

TimeStepDiffDFSPH::TimeStepDiffDFSPH():
  m_simulationData()
{
  m_simulationData.init();
  m_counter = 0;
  step_count = 0;
  m_iterationsV = 0;
  m_enableDivergenceSolver = true;
  m_use_MLS_pressure = false;
  m_use_symmetric_formula = false;
  m_use_strong_coupling = false;

  m_use_pressure_warmstart = true;
  m_use_divergence_warmstart = true;

  m_is_trajectory_finished_callback = false;
  m_is_in_new_trajectory = false;
  m_is_debug = false;
  m_optimize_rotation = true;
  m_maxIterationsV = 100;
  m_maxErrorV = static_cast<Real>(0.1);

  m_enable_count_neighbor_dof = false;

  Simulation *sim = Simulation::getCurrent();
  const unsigned int nModels = sim->numberOfFluidModels();
  for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
    {
      FluidModel *model = sim->getFluidModel(fluidModelIndex);
      model->addField({ "factor", FieldType::Scalar, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getFactor(fluidModelIndex, i); } });
      model->addField({ "advected density", FieldType::Scalar, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getDensityAdv(fluidModelIndex, i); } });
      model->addField({ "kappa", FieldType::Scalar, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getKappa(fluidModelIndex, i); }, true });
      model->addField({ "kappa_v", FieldType::Scalar, [this, fluidModelIndex](const unsigned int i) -> Real* { return &m_simulationData.getKappaV(fluidModelIndex, i); }, true });

#ifdef BACKWARD
      model->addField({ "sum_grad_p_k", FieldType::Vector3, [this, fluidModelIndex](const unsigned int i) -> Vector3r* { return &m_simulationData.get_sum_grad_p_k(fluidModelIndex, i); } });

      // For now we assume there is only one fluid, and one rigidbody (with index 1)
      m_simulationData.get_total_mass_fluid() = model->numActiveParticles() * model->getMass(0);
      // Note: cannot access boundary model here:
      //BoundaryModel_Akinci2012 *bm = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(1));
      //m_simulationData.get_total_mass_rb() = bm->getRigidBodyObject()->getMass();
#endif
    }
}

TimeStepDiffDFSPH::~TimeStepDiffDFSPH()
{
  Simulation *sim = Simulation::getCurrent();
  const unsigned int nModels = sim->numberOfFluidModels();
  for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
    {
      FluidModel *model = sim->getFluidModel(fluidModelIndex);
      model->removeFieldByName("factor");
      model->removeFieldByName("advected density");
      model->removeFieldByName("kappa");
      model->removeFieldByName("kappa_v");
#ifdef BACKWARD
      model->removeFieldByName("sum_grad_p_k");
#endif
    }

}

void TimeStepDiffDFSPH::initParameters()
{
  TimeStep::initParameters();

  SOLVER_ITERATIONS_V = createNumericParameter("iterationsV", "Iterations (divergence)", &m_iterationsV);
  setGroup(SOLVER_ITERATIONS_V, "DiffDFSPH");
  setDescription(SOLVER_ITERATIONS_V, "Iterations required by the divergence solver.");
  getParameter(SOLVER_ITERATIONS_V)->setReadOnly(true);

  MAX_ITERATIONS_V = createNumericParameter("maxIterationsV", "Max. iterations (divergence)", &m_maxIterationsV);
  setGroup(MAX_ITERATIONS_V, "DiffDFSPH");
  setDescription(MAX_ITERATIONS_V, "Maximal number of iterations of the divergence solver.");
  static_cast<NumericParameter<unsigned int>*>(getParameter(MAX_ITERATIONS_V))->setMinValue(1);

  MAX_ERROR_V = createNumericParameter("maxErrorV", "Max. divergence error(%)", &m_maxErrorV);
  setGroup(MAX_ERROR_V, "DiffDFSPH");
  setDescription(MAX_ERROR_V, "Maximal divergence error (%).");
  static_cast<RealParameter*>(getParameter(MAX_ERROR_V))->setMinValue(static_cast<Real>(1e-6));

  USE_DIVERGENCE_SOLVER = createBoolParameter("enableDivergenceSolver", "Enable divergence solver", &m_enableDivergenceSolver);
  setGroup(USE_DIVERGENCE_SOLVER, "DiffDFSPH");
  setDescription(USE_DIVERGENCE_SOLVER, "Turn divergence solver on/off.");

  USE_PRESSURE_WARMSTART = createBoolParameter("use pressure warmstart", "use pressure warmstart", &m_use_pressure_warmstart);
  setGroup(USE_PRESSURE_WARMSTART, "DiffDFSPH");

  USE_DIV_WARMSTART = createBoolParameter("use divergence warmstart", "use divergence warmstart", &m_use_divergence_warmstart);
  setGroup(USE_DIV_WARMSTART, "DiffDFSPH");

  USE_SYMMETRIC_FORMULA = createBoolParameter("useSymmetricFormula", "use symmetric formula", &m_use_symmetric_formula);
  setGroup(USE_SYMMETRIC_FORMULA, "DiffDFSPH");

  USE_MLS_PRESSURE = createBoolParameter("use mls pressure", "use mls pressure", &m_use_MLS_pressure);
  setGroup(USE_MLS_PRESSURE, "DiffDFSPH");

  USE_STRONG_COUPLING = createBoolParameter("use strong coupling", "use strong coupling", &m_use_strong_coupling);
  setGroup(USE_STRONG_COUPLING, "DiffDFSPH");
#ifdef BACKWARD
  // =====================================================

  DEBUG = createBoolParameter("is debug", "is debug", &m_is_debug);
  setGroup(DEBUG, "DiffDFSPH");

  ENABLE_COUNT_NEIGHBOR_DOF = createBoolParameter("count neighbor dof", "count neighbor dof", &m_enable_count_neighbor_dof);
  setGroup(ENABLE_COUNT_NEIGHBOR_DOF, "DiffDFSPH");

  N_ITER = createNumericParameter("iteration", "iteration", &m_simulationData.get_opt_iter());
  setGroup(N_ITER, "DiffDFSPH");
  getParameter(N_ITER)->setReadOnly(true);

  LOSS = createNumericParameter("loss", "loss", &m_simulationData.get_loss());
  setGroup(LOSS, "DiffDFSPH");
  getParameter(LOSS)->setReadOnly(true);

  LR = createNumericParameter("learning rate", "learning rate", &m_simulationData.get_learning_rate());
  setGroup(LR, "DiffDFSPH");

  // =====================================================
  TARGET_TIME = createNumericParameter("target time", "target time", &m_simulationData.get_target_time());
  setGroup(TARGET_TIME, "DiffDFSPH");
  static_cast<RealParameter*>(getParameter(MAX_ERROR_V))->setMinValue(static_cast<Real>(1e-6));

  UNIFORM_ACC_RB_TIME = createNumericParameter("uniform_acc_rb_time", "uniform_acc_rb_time", &m_simulationData.get_uniform_accelerate_rb_time_at_beginning());
  setGroup(UNIFORM_ACC_RB_TIME, "DiffDFSPH");
  static_cast<RealParameter*>(getParameter(MAX_ERROR_V))->setMinValue(static_cast<Real>(1e-6));

  TARGET_X = createVectorParameter("target x rb", "target x rb", 3, m_simulationData.get_target_x(1).data() );
  setGroup(TARGET_X, "DiffDFSPH");

  //FINAL_X_RB = createVectorParameter("final x rb", "final x rb", 3, m_simulationData.get_final_x_rb(1).data() );
  //setGroup(FINAL_X_RB, "DiffDFSPH");
  //getParameter(FINAL_X_RB)->setReadOnly(true);

  CURRENT_X_RB = createVectorParameter("current x", "current x", 3, m_simulationData.get_current_x_rb(1).data());
  setGroup(CURRENT_X_RB, "DiffDFSPH");
  getParameter(CURRENT_X_RB)->setReadOnly(true);

  //TARGET_QUATERNION = createVectorParameter("target quaternion", "target quaternion", 4, get_target_quaternion().data() );
  //setGroup(TARGET_QUATERNION, "DiffDFSPH");

  //FINAL_QUATERNION_RB = createVectorParameter("final quaternion rb", "final quaternion rb", 4, get_final_quaternion_rb().data() );
  //setGroup(FINAL_QUATERNION_RB, "DiffDFSPH");
  //getParameter(FINAL_QUATERNION_RB)->setReadOnly(true);

  INIT_V_RB = createVectorParameter("init v rb", "init v rb", 3, m_simulationData.get_init_v_rb(1).data() );
  setGroup(INIT_V_RB, "DiffDFSPH");

  CURRENT_V_RB = createVectorParameter("current v rb", "current v rb", 3,m_simulationData.get_current_v_rb(1).data() );
  setGroup(CURRENT_V_RB, "DiffDFSPH");
  getParameter(CURRENT_V_RB)->setReadOnly(true);
  // ====================================================

  //GRAD_INIT_V_RB = createVectorParameter("grad init v", "grad init v", 3, m_simulationData.get_grad_init_v_rb().data() );
  //setGroup(GRAD_INIT_V_RB, "DiffDFSPH");
  //getParameter(GRAD_INIT_V_RB)->setReadOnly(true);

  OPTIMIZE_ROTATION = createBoolParameter("optimize rotation", "optimize rotation", &m_optimize_rotation);
  setGroup(OPTIMIZE_ROTATION, "DiffDFSPH");

  //TARGET_ANGLE = createVectorParameter("target angle (in degree)", "target angle (in degree)", 3, m_simulationData.get_target_angle_in_degree().data() );
  //setGroup(TARGET_ANGLE, "DiffDFSPH");

  //FINAL_ANGLE_RB = createVectorParameter("final angle (in degree)", "final angle rb (in degree)", 3, m_simulationData.get_final_angle_rb_in_degree().data() );
  //setGroup(FINAL_ANGLE_RB, "DiffDFSPH");
  //getParameter(FINAL_ANGLE_RB)->setReadOnly(true);

  INIT_OMEGA_RB = createVectorParameter("init omega rb", "init omega rb", 3, m_simulationData.get_init_omega_rb(1).data() );
  setGroup(INIT_OMEGA_RB, "DiffDFSPH");

  CURRENT_OMEGA_RB = createVectorParameter("current omega rb", "current omega rb", 3,m_simulationData.get_current_omega_rb(1).data() );
  setGroup(CURRENT_OMEGA_RB, "DiffDFSPH");
  getParameter(CURRENT_OMEGA_RB)->setReadOnly(true);

  TOTAL_MASS_RB = createNumericParameter("total mass rb", "total mass rb", &m_simulationData.get_total_mass_rb());
  setGroup(TOTAL_MASS_RB, "DiffDFSPH");
  getParameter(TOTAL_MASS_RB)->setReadOnly(true);

  TOTAL_MASS_FLUID= createNumericParameter("total mass fluid", "total mass fluid", &m_simulationData.get_total_mass_fluid());
  setGroup(TOTAL_MASS_FLUID, "DiffDFSPH");
  getParameter(TOTAL_MASS_FLUID)->setReadOnly(true);
  //GRAD_INIT_OMEGA_RB = createVectorParameter("grad init omega", "grad init omega", 3, m_simulationData.get_grad_init_omega_rb().data() );
  //setGroup(GRAD_INIT_OMEGA_RB, "DiffDFSPH");
  //getParameter(GRAD_INIT_OMEGA_RB)->setReadOnly(true);


#endif

}

void TimeStepDiffDFSPH::countNeighborDOF()  
{
	Simulation *sim = Simulation::getCurrent();
	TimeManager *tm = TimeManager::getCurrent();
	const Real t = tm->getTime();
	const unsigned int nModels = sim->numberOfFluidModels();
	const unsigned int nFluids = nModels;
	//LOG_INFO << "Time " << t << '\n';
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();
	const unsigned int nPointSet = sim->numberOfPointSets();
	std::vector<std::vector<int>> rigid_particle_has_fluid_neighbor_status;
	rigid_particle_has_fluid_neighbor_status.resize(nPointSet);

	unsigned int fluid_and_rigid_particle_count = 0;
  this->m_simulationData.num_1ring_fluid_particle = 0; 

  // First, initalize the rigid_particle_has_fluid_neighbor_status 
	for (unsigned int i = nFluids; i < nPointSet; i++)
	{
		BoundaryModel_Akinci2012 *bm = static_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModelFromPointSet(i));
    if(bm->getRigidBodyObject()->isDynamic() || bm->getRigidBodyObject()->isAnimated())
    {
      // std::cout << i << '\t' << bm->numberOfParticles() << '\n';
      rigid_particle_has_fluid_neighbor_status[i].resize(bm->numberOfParticles(), 0);
    }
	}

  // Then, traverse all fluid particles, and check if they have dynamic neighbor rigid particles 
	for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels; fluidModelIndex++)
	{
		FluidModel *model = sim->getFluidModel(fluidModelIndex);
		const unsigned int numFluidParticles = model->numActiveParticles();
    // std::cout << "fluidModel " << fluidModelIndex << " start " <<count << "\t" <<numParticles << '\n';
		for (int i = 0; i < numFluidParticles;i++)
		{
			for (unsigned int pid = nFluids; pid < nPointSet; pid++)
			{
				BoundaryModel_Akinci2012 *bm = static_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModelFromPointSet(pid));
        if(!bm->getRigidBodyObject()->isDynamic() && !bm->getRigidBodyObject()->isAnimated())
          continue;

				unsigned int neighborRigidParticleNum = sim->numberOfNeighbors(fluidModelIndex, pid, i);
				if(neighborRigidParticleNum >  0) // Means this fluid partilce has one-ring neighbor rigid particle 
        {
          fluid_and_rigid_particle_count++;
          this->m_simulationData.num_1ring_fluid_particle ++; // Which is equal to: this fluid particle is the one-rigid neighbor particle of rigid body 
        }

        // Traverse the neighbor rigid particle of fluid particle {i}
				for (unsigned int j = 0; j < neighborRigidParticleNum;j++)
				{
					const unsigned int neighborRigidParticleIndex = sim->getNeighbor(fluidModelIndex, pid, i, j);
					if (rigid_particle_has_fluid_neighbor_status[pid][neighborRigidParticleIndex] == 0) // if this rigid particle has not been marked with "has fluid neighbor" yet
					{
						rigid_particle_has_fluid_neighbor_status[pid][neighborRigidParticleIndex] = 1;
						fluid_and_rigid_particle_count++;
					}
				}
			}
		}
    // std::cout << "fluidModel " << fluidModelIndex << " end " <<count << '\n';
	}
	//LOG_INFO << "Particle num:\t"<< fluid_and_rigid_particle_count << '\n';
  if(fluid_and_rigid_particle_count > 10)
  {
    INCREASE_COUNTER("involved particle num", fluid_and_rigid_particle_count);
  }
}


void TimeStepDiffDFSPH::beginStep()
{
  step_count ++;
  Simulation *sim = Simulation::getCurrent();
  if (sim->isDebug())
  {
    LOG_INFO << GreenHead() << " timestep = " << step_count << GreenTail();
    total_force = Vector3r::Zero();
  }
  TimeManager *tm = TimeManager::getCurrent();
  if(tm->getTime() < 1e-6)
  {
    // At the beginning of simulation,
    // 1. set some parameters
    // 2. record some variables for later use
    m_simulationData.get_total_mass_rb() = 0.; 

    forall_dynamic_boundary_model_Akinci12(
      m_simulationData.get_total_mass_rb() += bm_i->getRigidBodyObject()->getMass();
      m_simulationData.get_init_rb_rotation(bm_index) = bm_i->getRigidBodyObject()->getRotation();
      LOG_INFO << CyanHead() << "[ rb" << bm_index <<"] set init v rb = " << m_simulationData.get_init_v_rb(bm_index).transpose() << CyanTail();
      LOG_INFO << CyanHead() << "[ rb" << bm_index <<"] set init omega rb = " << m_simulationData.get_init_omega_rb(bm_index).transpose() << CyanTail();
  );
    m_simulationData.increase_opt_iter();
  }

  forall_dynamic_boundary_model_Akinci12(

    if(sim->useReleaseRigidBodyMode()) // add logic here to first release the second rigid ball 
    {
      // if boundary model index is 1, it is the act rigid body while other rigidbodies are passive 
      // and we'd like to keep the act rigid body not moving before {uniform_acc_rb_time} to let 
      // other passive rigid bodies move first with a uniform accleration. 
      // (Note that this logic can be scalable by changing "bm_index == 1" to "bm_index in list_of_act_rb")
      if(bm_index == 1) 
      {
        Real uniform_acc_rb_time = m_simulationData.get_uniform_accelerate_rb_time_at_beginning();
        if(tm->getTime() <= uniform_acc_rb_time + TimeManager::getCurrent()->getTimeStepSize())
        {
          if (uniform_acc_rb_time > 1e-3)
          {
            bm_i->getRigidBodyObject()->setIsAnimated(true); // to make sure the rigid body does not move with physics
          }
              //bm_i->getRigidBodyObject()->setVelocity(factor * m_simulationData.get_init_v_rb(bm_index));
          bm_i->getRigidBodyObject()->setVelocity(Vector3r::Zero());
          bm_i->getRigidBodyObject()->setAngularVelocity(Vector3r::Zero());
        }
        else // Now we release this acting rigid body with its init velocity and angular velocity
        {
          bm_i->getRigidBodyObject()->setVelocity(m_simulationData.get_init_v_rb(bm_index));
          bm_i->getRigidBodyObject()->setAngularVelocity(m_simulationData.get_init_omega_rb(bm_index));
          bm_i->getRigidBodyObject()->setIsAnimated(false);
        }
      }
    }
    else  // for othe passive rigid bodies, they move first with a uniform accelaration time 
    {
      Real uniform_acc_rb_time = m_simulationData.get_uniform_accelerate_rb_time_at_beginning();
       if(tm->getTime() <= uniform_acc_rb_time + TimeManager::getCurrent()->getTimeStepSize())
      {
        Real factor = 1. ;
        if (uniform_acc_rb_time > 1e-3)
        {
          factor = (tm->getTime() / uniform_acc_rb_time) >  1. ? 1. : (tm->getTime() / uniform_acc_rb_time); // min
          bm_i->getRigidBodyObject()->setIsAnimated(true); // to make sure the rigid body does not move with physics
        }
        bm_i->getRigidBodyObject()->setVelocity(factor * m_simulationData.get_init_v_rb(bm_index));
        bm_i->getRigidBodyObject()->setAngularVelocity(factor * m_simulationData.get_init_omega_rb(bm_index));
      }
      else
        {
          bm_i->getRigidBodyObject()->setIsAnimated(false);
        }

    }

  );
}

void TimeStepDiffDFSPH::endStep()
{
  Simulation *sim = Simulation::getCurrent();
  if(sim->isDebug())
  {
      LOG_INFO << Utilities::YellowHead() <<  "net force = " << total_force.transpose() << Utilities::YellowTail() << "\n";
      total_force = Vector3r::Zero();
  }
  TimeManager *tm = TimeManager::getCurrent();
  Real uniform_acc_rb_time = m_simulationData.get_uniform_accelerate_rb_time_at_beginning();

  // End of this iteration of simulation
  // 1. get final state: position & rotation
  // 2. accumulate_and_reset_gradient & perform chain rule to get final gradient
  // 3. set need_step_callback be true
  // 4. output some useful log
  if(tm->getTime() >= m_simulationData.get_target_time() + uniform_acc_rb_time)
  {
    LOG_INFO << GreenHead() << "[ iter "<< m_simulationData.get_opt_iter() <<"] end time =" << tm->getTime() << GreenTail() << "\n";

    forall_dynamic_boundary_model_Akinci12(
      auto final_x_rb = bm_i->getRigidBodyObject()->getPosition();
      // https://stackoverflow.com/questions/31589901/euler-to-quaternion-quaternion-to-euler-using-eigen
      //auto q_ref = m_simulationData.get_init_rb_rotation();
      //auto final_quaternion_rb =  bm->getRigidBodyObject()->getRotation() * q_ref.inverse(); // new rotation = q_ref x delta_q
      auto final_quaternion_rb =  bm_i->getRigidBodyObject()->getRotation(); // new rotation = q_ref x delta_q

      //m_simulationData.get_final_x_rb() = final_x_rb;
      //m_simulationData.get_final_quaternion_rb() = final_quaternion_rb;
      LOG_INFO << GreenHead() << " total timestep of a trajectory = " << this->step_count << GreenTail();

      m_is_trajectory_finished_callback = true;
      m_simulationData.clear_custom_log_message();

      // Note:  sim->reset() & this->reset() is not enough to reset all states
      //sim->getSimulatorBase()->reset(); // which should be done in timestep callBack
      if(m_is_in_new_trajectory)
      {
        LOG_INFO << GreenHead() << "[ iter "<< m_simulationData.get_opt_iter() <<"] ======= in LBFGS line search =======" << GreenTail();
      }

      LOG_INFO << GreenHead() << "[ rb " << bm_index << "][ init ] init x =" << m_simulationData.get_init_x(bm_index).transpose() << GreenTail();
      LOG_INFO << GreenHead() << "[ rb " << bm_index << "][ target ] target x =" << m_simulationData.get_target_x(bm_index).transpose() << GreenTail();
      LOG_INFO << GreenHead() << "[ rb " << bm_index << "][ iter "<< m_simulationData.get_opt_iter() <<"] final x_rb ="
        << bm_i->getRigidBodyObject()->getPosition().transpose() << GreenTail() << "\n";
      LOG_INFO << GreenHead() << "[ rb " << bm_index << "][ target ] target angle (only meaningful in 2d or 2.5d) ="
        << m_simulationData.get_target_angle_in_degree(bm_index).transpose() << GreenTail();
      LOG_INFO << GreenHead() << "[ rb " << bm_index << "][ iter "<< m_simulationData.get_opt_iter() <<"] final angle_rb (in degree) (only meaningful in 2d or 2.5d) ="
        << bm_i->getRigidBodyObject()->getAngleDistanceInRadian().transpose() / M_PI * 180. << GreenTail() << "\n";

      LOG_INFO << GreenHead() << "[ rb " << bm_index << "][ target ] target quaternion =" << get_target_quaternion_vec4(bm_index).transpose() << GreenTail();
      LOG_INFO << GreenHead() << "[ rb " << bm_index << "][ iter "<< m_simulationData.get_opt_iter() <<"] final quaternion_rb ="
        << bm_i->get_quaternion_rb_vec4().transpose() << GreenTail() << "\n";
    );
  }
  else {
      m_is_trajectory_finished_callback = false;
  }
}

// TODO: move this part to a common rigidbody gradient manager 
void TimeStepDiffDFSPH::backwardPerStep()
{
  Simulation *sim = Simulation::getCurrent();
  TimeManager *tm = TimeManager::getCurrent();
  const Real h = tm->getTimeStepSize();

  START_TIMING("chainRule");

  if(sim->numberOfFluidModels() > 0 )
  {
    forall_dynamic_boundary_model_Akinci12(
      if(false == bm_i->getRigidBodyObject()->isAnimated())
      {
        bm_i->accumulate_and_reset_gradient();
      if(sim->useRigidGradientManager())
        bm_i->update_rigid_body_gradient_manager(); 
      else
        bm_i->perform_chain_rule(h, m_optimize_rotation);
      }
  }
      //m_simulationData.get_last_rotation() = bm->getRigidBodyObject()->getRotation();
      //m_simulationData.get_final_angle_rb_in_radian() += h * bm->getRigidBodyObject()->getAngularVelocity();
      //m_simulationData.get_final_angle_rb_in_degree() = m_simulationData.get_final_angle_rb_in_radian() / M_PI * 180.;
  m_simulationData.get_current_omega_rb(bm_index) = bm_i->getRigidBodyObject()->getAngularVelocity();
  m_simulationData.get_current_v_rb(bm_index) = bm_i->getRigidBodyObject()->getVelocity();
  m_simulationData.get_current_x_rb(bm_index) = bm_i->getRigidBodyObject()->getPosition();

  //FIXME: one step left: the position of rigidbody is not updated at this moment
);
  STOP_TIMING_AVG("chainRule");

}

void TimeStepDiffDFSPH::step()
{
  START_TIMING("totalTimestep");
  Simulation *sim = Simulation::getCurrent();
  TimeManager *tm = TimeManager::getCurrent();
#ifdef BACKWARD
  beginStep();
#endif

  const Real h = tm->getTimeStepSize();
  const unsigned int nModels = sim->numberOfFluidModels();

  START_TIMING("neighborSearch");
  performNeighborhoodSearch(); // don't need gradient
  STOP_TIMING_DETAIL("neighborSearch");
  
  if (m_enable_count_neighbor_dof)
    countNeighborDOF();

  // To begin with, we use Akinci2012 way to do coupling
  START_TIMING("computeDensity");
  for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels;
    fluidModelIndex ++)
    computeDensities(fluidModelIndex); // need gradient
  STOP_TIMING_AVG;

  START_TIMING("computeDFSPHFactor");
  for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels;
    fluidModelIndex ++)
    {
      computeDFSPHFactor(fluidModelIndex); 
    }
  STOP_TIMING_AVG;

  if(m_enableDivergenceSolver)
  {
    START_TIMING("divergenceSolve");
    // A key point lies in how to differentiate this part
    divergenceSolve(); // need gradient
    STOP_TIMING_AVG("divergenceSolve");
  }
else
  m_iterationsV = 0 ;

  START_TIMING("clearAccelerations");
  for (unsigned int fluidModelIndex = 0; fluidModelIndex < nModels;
    fluidModelIndex ++)
    clearAccelerations(fluidModelIndex);
  STOP_TIMING_AVG;

  START_TIMING("computeNonPressureForces");
  sim->computeNonPressureForces();
  STOP_TIMING_DETAIL("computeNonPressureForces");

  ////use an array to store all the timesteps
  ////m_simulationData.recordTimeStep(h); // TODO
  
  START_TIMING("updateTimeStepSize");
  sim->updateTimeStepSize();
  STOP_TIMING_AVG;

  ////  do velocity advect
  START_TIMING("velocity Adect");
  for(unsigned int m = 0; m < nModels; m++)
    {
      FluidModel *fm = sim->getFluidModel(m);
      const unsigned int numParticles = fm->numActiveParticles();
      #pragma omp parallel default(shared)
      {
        #pragma omp for schedule(static)
        for (int i=0; i < (int)numParticles; i ++) {
          if (fm->getParticleState(i) == ParticleState::Active)
          {
            Vector3r &vel = fm->getVelocity(i);
            vel += h * fm->getAcceleration(i); // need grad
          }
        }
      }
    }
  STOP_TIMING_AVG;

  // For now velocity is v_n+1, but position is x_n, not x_n+1
  
  START_TIMING("pressureSolve");
  pressureSolve();  // need grad
  STOP_TIMING_AVG;
  
  START_TIMING("velocity Adect");

  for (unsigned int m = 0; m < nModels; m ++) {
    FluidModel * fm  = sim->getFluidModel(m);

    const unsigned int numParticles = fm -> numActiveParticles();
    #pragma omp parallel default(shared)
    {
      #pragma omp for schedule(static)
      for (int i = 0; i < (int)numParticles; i++)
        {
          if (fm->getParticleState(i) == ParticleState::Active)
          {
            Vector3r &xi = fm->getPosition(i);
            const Vector3r &vi = fm->getVelocity(i);
            xi += h * vi; // x_n+1 = x_n + h* v_n+1

          }
        }
    }

  }
  STOP_TIMING_AVG;

  START_TIMING("emit");
  sim->emitParticles();
  STOP_TIMING_AVG;
  START_TIMING("animate");
  sim->animateParticles();
  STOP_TIMING_AVG;

  // compute new time
  tm->setTime(tm->getTime() + h);

#ifdef BACKWARD
  backwardPerStep();
  endStep();
#endif
  STOP_TIMING_AVG("totalTimestep");
}

void TimeStepDiffDFSPH::pressureSolve()
{
  const Real h = TimeManager::getCurrent()->getTimeStepSize();
  const Real h2 = h*h;
  const Real invH = static_cast<Real>(1.0) / h;
  const Real invH2 = static_cast<Real>(1.0) / h2;
  Simulation *sim = Simulation::getCurrent();
  const unsigned int nFluids = sim->numberOfFluidModels();
  if(nFluids == 0)
    return; 

  if(m_use_strong_coupling)
    sim->getSimulatorBase()->getBoundarySimulator()->getRigidContactSolver()->beforeIterationInitialize(); 

  if(m_use_pressure_warmstart)
  {
    for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
      warmstartPressureSolve(fluidModelIndex);

    if(m_use_strong_coupling)
    {
      //sim->getSimulatorBase()->getBoundarySimulator()->updateVelocity(); 
      // TODO: add one step iteration of rigid contact solver here 
       sim->getSimulatorBase()->getBoundarySimulator()->getRigidContactSolver()->step(); 
    }
  }

  //////////////////////////////////////////////////////////////////////////
  // Compute rho_adv
  //////////////////////////////////////////////////////////////////////////
  for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
    {
      FluidModel *model = sim->getFluidModel(fluidModelIndex);
      const Real density0 = model->getDensity0();
      const int numParticles = (int)model->numActiveParticles();
      #pragma omp parallel default(shared)
      {
        #pragma omp for schedule(static)
        for (int i = 0; i < numParticles; i++)
          {
            computeDensityAdv(fluidModelIndex, i, numParticles, h, density0);
            //m_simulationData.getFactor(fluidModelIndex, i) *= invH2;
          }
      }
    }

  m_iterations = 0;

  //////////////////////////////////////////////////////////////////////////
  // Start solver
  //////////////////////////////////////////////////////////////////////////

  Real avg_density_err = 0.0;
  bool chk = false;
  int n_fluid_neighbors = 0;
  Real last_rigid_avg_density_err = 0.0; 

  while ((!chk || (m_iterations < m_minIterations)) && (m_iterations < m_maxIterations))
    {
      chk = true;
      for (unsigned int i = 0; i < nFluids; i++)
        {
          FluidModel *model = sim->getFluidModel(i);
          const Real density0 = model->getDensity0();

          avg_density_err = 0.0;
          n_fluid_neighbors += pressureSolveIteration(i, avg_density_err);

          Real rigid_avg_density_err = 0.;
          Real rigid_eta = 0.; 
          bool stopRigidContactSolver = true;
          if(m_use_strong_coupling)
          {
            rigid_eta = sim->getSimulatorBase()->getBoundarySimulator()->getRigidContactSolver()->get_density_err_thresh();
            //sim->getSimulatorBase()->getBoundarySimulator()->updateVelocity(); 
            rigid_avg_density_err = sim->getSimulatorBase()->getBoundarySimulator()->getRigidContactSolver()->step(); 
            LOG_INFO << Utilities::RedHead() << "rigid_avg_density_err = " << rigid_avg_density_err << Utilities::RedTail();
            stopRigidContactSolver = (rigid_avg_density_err <= rigid_eta) || 
                          (fabs(last_rigid_avg_density_err - rigid_avg_density_err) < 1e-6);
          }

          // Maximal allowed density fluctuation
          const Real eta = m_maxError * static_cast<Real>(0.01) * density0;  // maxError is given in percent
          chk = chk && (avg_density_err <= eta) && stopRigidContactSolver;

          last_rigid_avg_density_err = rigid_avg_density_err; 
        }

      m_iterations++;
    }
  //TODO: add RigidContactSolver afterIterationSolve to handle friction
  

  if(sim->isDebug())
    LOG_INFO << CyanHead() << "fluid_particle_neighbor_count = " << n_fluid_neighbors << CyanTail();

  INCREASE_COUNTER("DFSPH - iterations", static_cast<Real>(m_iterations));

  if(m_use_pressure_warmstart)
  {
    for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
      {
        FluidModel *model = sim->getFluidModel(fluidModelIndex);
        const int numParticles = (int)model->numActiveParticles();

        //////////////////////////////////////////////////////////////////////////
        // Multiply by h^2, the time step size has to be removed
        // to make the stiffness value independent
        // of the time step size
        //////////////////////////////////////////////////////////////////////////
        for (int i = 0; i < numParticles; i++)
          m_simulationData.getKappa(fluidModelIndex, i) *= h2;
      }
  }
}

void TimeStepDiffDFSPH::divergenceSolve()
{
  //////////////////////////////////////////////////////////////////////////
  // Init parameters
  //////////////////////////////////////////////////////////////////////////

  const Real h = TimeManager::getCurrent()->getTimeStepSize();
  const Real invH = static_cast<Real>(1.0) / h;
  Simulation *sim = Simulation::getCurrent();
  const unsigned int maxIter = m_maxIterationsV;
  const Real maxError = m_maxErrorV;
  const unsigned int nFluids = sim->numberOfFluidModels();
  if(nFluids == 0)
    return; 

  if(m_use_strong_coupling)
    sim->getSimulatorBase()->getBoundarySimulator()->getRigidContactSolver()->beforeIterationInitialize(); 

  if(m_use_divergence_warmstart)
  {
  for(unsigned int fluidModelIndex =0; fluidModelIndex < nFluids; fluidModelIndex++)
    warmstartDivergenceSolve(fluidModelIndex);
    if(m_use_strong_coupling)
    {
      //sim->getSimulatorBase()->getBoundarySimulator()->updateVelocity(); 
      sim->getSimulatorBase()->getBoundarySimulator()->getRigidContactSolver()->step(); 
    }
  }

  //////////////////////////////////////////////////////////////////////////
  // Compute velocity of density change
  //////////////////////////////////////////////////////////////////////////
  for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
    {
      FluidModel *model = sim->getFluidModel(fluidModelIndex);
      const int numParticles = (int)model->numActiveParticles();

      #pragma omp parallel default(shared)
      {
        #pragma omp for schedule(static)
        for (int i = 0; i < numParticles; i++)
          {
            computeDensityChange(fluidModelIndex, i, h);
            //m_simulationData.getFactor(fluidModelIndex, i) *= invH;
          }
      }
    }

  m_iterationsV = 0;

  //////////////////////////////////////////////////////////////////////////
  // Start solver
  //////////////////////////////////////////////////////////////////////////

  Real avg_density_err = 0.0;
  bool chk = false;
  Real last_rigid_avg_density_err = 0.0;

  while ((!chk || (m_iterationsV < 1)) && (m_iterationsV < maxIter))
    {
      chk = true;
      for (unsigned int i = 0; i < nFluids; i++)
        {
          FluidModel *model = sim->getFluidModel(i);
          const Real density0 = model->getDensity0();

          avg_density_err = 0.0;
          divergenceSolveIteration(i, avg_density_err);

          Real rigid_avg_density_err = 0.;
          Real rigid_eta = 0.; 
          bool stopRigidContactSolver = true;
          if(m_use_strong_coupling)
          {
            rigid_eta = sim->getSimulatorBase()->getBoundarySimulator()->getRigidContactSolver()->get_density_err_thresh();
            //sim->getSimulatorBase()->getBoundarySimulator()->updateVelocity(); 
            rigid_avg_density_err = sim->getSimulatorBase()->getBoundarySimulator()->getRigidContactSolver()->step(); 
            LOG_INFO << Utilities::RedHead() << "rigid_avg_density_err = " << rigid_avg_density_err << Utilities::RedTail();
            stopRigidContactSolver = (rigid_avg_density_err <= rigid_eta) || 
                          (fabs(last_rigid_avg_density_err - rigid_avg_density_err) < 1e-6);
          }

          // Maximal allowed density fluctuation
          // use maximal density error divided by time step size
          const Real eta = (static_cast<Real>(1.0) / h) * maxError * static_cast<Real>(0.01) * density0;  // maxError is given in percent
          chk = chk && (avg_density_err <= eta) && stopRigidContactSolver;

          last_rigid_avg_density_err = rigid_avg_density_err; 
        }

      m_iterationsV++;
    }

  INCREASE_COUNTER("DFSPH - iterationsV", static_cast<Real>(m_iterationsV));

  //////////////////////////////////////////////////////////////////////////
  // Multiply by h, the time step size has to be removed
  // to make the stiffness value independent
  // of the time step size
  //////////////////////////////////////////////////////////////////////////
  if(m_use_divergence_warmstart)
  {
    for (unsigned int fluidModelIndex = 0; fluidModelIndex < nFluids; fluidModelIndex++)
    {
      FluidModel *model = sim->getFluidModel(fluidModelIndex);
      const int numParticles = (int)model->numActiveParticles();

      for (int i = 0; i < numParticles; i++)
        m_simulationData.getKappaV(fluidModelIndex, i) *= h;
    }
  }
}

void TimeStepDiffDFSPH::computeDFSPHFactor(const unsigned int fluidModelIndex)
{
  //////////////////////////////////////////////////////////////////////////
  // Init parameters
  //////////////////////////////////////////////////////////////////////////

  Simulation *sim = Simulation::getCurrent();
  const unsigned int nFluids = sim->numberOfFluidModels();
  const unsigned int nBoundaries = sim->numberOfBoundaryModels();
  FluidModel *model = sim->getFluidModel(fluidModelIndex);
  const int numParticles = (int) model->numActiveParticles();

  #pragma omp parallel default(shared)
  {
    //////////////////////////////////////////////////////////////////////////
    // Compute pressure stiffness denominator
    //////////////////////////////////////////////////////////////////////////

    #pragma omp for schedule(static)
    for (int i = 0; i < numParticles; i++)
      {
        //////////////////////////////////////////////////////////////////////////
        // Compute gradient dp_i/dx_j * (1/k)  and dp_j/dx_j * (1/k)
        //////////////////////////////////////////////////////////////////////////
        const Vector3r &xi = model->getPosition(i);
        Real sum_grad_p_k = 0.0;
        Vector3r grad_p_i;
        grad_p_i.setZero();

        //////////////////////////////////////////////////////////////////////////
        // Fluid
        //////////////////////////////////////////////////////////////////////////
        forall_fluid_neighbors(
          const Vector3r grad_p_j = -fm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
          sum_grad_p_k += grad_p_j.squaredNorm();
          grad_p_i -= grad_p_j;
        );

        //////////////////////////////////////////////////////////////////////////
        // Boundary
        //////////////////////////////////////////////////////////////////////////
        if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
        {
          forall_boundary_neighbors(
            const Vector3r grad_p_j = -bm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
          grad_p_i -= grad_p_j;
        );
        }

      else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
        {
          forall_density_maps(
              grad_p_i -= gradRho;
            );
        }
      else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
        {
          forall_volume_maps(
            const Vector3r grad_p_j = -Vj * sim->gradW(xi - xj);
          grad_p_i -= grad_p_j;
        );
        }

#ifdef BACKWARD
        Vector3r& sum_grad = m_simulationData.get_sum_grad_p_k(fluidModelIndex, i);
        sum_grad =  - grad_p_i;
#endif
        sum_grad_p_k += grad_p_i.squaredNorm();

        //////////////////////////////////////////////////////////////////////////
        // Compute pressure stiffness denominator
        //////////////////////////////////////////////////////////////////////////
        Real &factor = m_simulationData.getFactor(fluidModelIndex, i);
        if (sum_grad_p_k > m_eps)
          factor = -static_cast<Real>(1.0) / (sum_grad_p_k);
        else
          factor = 0.0;
      }
  }
}

void TimeStepDiffDFSPH::warmstartPressureSolve(const unsigned int fluidModelIndex)
{
	const Real h = TimeManager::getCurrent()->getTimeStepSize();
	const Real h2 = h*h;
	const Real invH = static_cast<Real>(1.0) / h;
	const Real invH2 = static_cast<Real>(1.0) / h2;
	Simulation *sim = Simulation::getCurrent();
	FluidModel *model = sim->getFluidModel(fluidModelIndex);
	const Real density0 = model->getDensity0();
	const int numParticles = (int)model->numActiveParticles();
	if (numParticles == 0)
		return;

	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();

	#pragma omp parallel default(shared)
	{
		//////////////////////////////////////////////////////////////////////////
		// Divide by h^2, the time step size has been removed in 
		// the last step to make the stiffness value independent 
		// of the time step size
		//////////////////////////////////////////////////////////////////////////
    #pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			//m_simulationData.getKappa(fluidModelIndex, i) = max(m_simulationData.getKappa(fluidModelIndex, i)*invH2, -static_cast<Real>(0.5) * density0*density0);
			computeDensityAdv(fluidModelIndex, i, numParticles, h, density0);
			if (m_simulationData.getDensityAdv(fluidModelIndex, i) > 1.0)
				//m_simulationData.getKappa(fluidModelIndex, i) = static_cast<Real>(0.5) * max(m_simulationData.getKappa(fluidModelIndex, i), static_cast<Real>(-0.00025)) * invH2;
        m_simulationData.getKappa(fluidModelIndex, i) = static_cast<Real>(0.5) *
            max(m_simulationData.getKappa(fluidModelIndex, i), static_cast<Real>(-0.00025))
            * invH2; 
			else
				m_simulationData.getKappa(fluidModelIndex, i) = 0.0;
		}

		//////////////////////////////////////////////////////////////////////////
		// Predict v_adv with external velocities
		////////////////////////////////////////////////////////////////////////// 

		#pragma omp for schedule(static)  
		for (int i = 0; i < numParticles; i++)
		{
			if (model->getParticleState(i) != ParticleState::Active)
			{
				m_simulationData.getKappa(fluidModelIndex, i) = 0.0;
				continue;
			}

			//if (m_simulationData.getDensityAdv(fluidModelIndex, i) > 1.0)
			{
				Vector3r &vel = model->getVelocity(i);
				const Real ki = m_simulationData.getKappa(fluidModelIndex, i);
        model->getPressure(i) = ki * model->getDensity(i);
				const Vector3r &xi = model->getPosition(i);

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				forall_fluid_neighbors(
					const Real kj = m_simulationData.getKappa(pid, neighborIndex);

					const Real kSum = (ki + fm_neighbor->getDensity0() / density0 * kj);
					if (fabs(kSum) > m_eps)
					{
						const Vector3r grad_p_j = -fm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
						vel -= h * kSum * grad_p_j;					// ki, kj already contain inverse density
					}
				)

				//////////////////////////////////////////////////////////////////////////
				// Boundary
				//////////////////////////////////////////////////////////////////////////
				if (fabs(ki) > m_eps)
				{
					if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
					{
						forall_boundary_neighbors(
							const Vector3r grad_p_j = -bm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
							const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
							vel += velChange;
              Vector3r force = -model->getMass(i) * velChange * invH;
							bm_neighbor->addForce(xj, force);
              if(bm_neighbor->getRigidBodyObject()->isDynamic()
                && false == bm_neighbor->getRigidBodyObject()->isAnimated())
              {
                total_force += force;
              }
						);
					}
        }
			}
		}
	}
}

int TimeStepDiffDFSPH::pressureSolveIteration(const unsigned int fluidModelIndex, Real &avg_density_err)
{

  Simulation *sim = Simulation::getCurrent();
  FluidModel *model = sim->getFluidModel(fluidModelIndex);
  const Real density0 = model->getDensity0();
  const int numParticles = (int)model->numActiveParticles();
  if (numParticles == 0)
    return 0;

  const unsigned int nFluids = sim->numberOfFluidModels();
  const unsigned int nBoundaries = sim->numberOfBoundaryModels();
  const Real h = TimeManager::getCurrent()->getTimeStepSize();
  const Real h2 = h * h;
  const Real invH = static_cast<Real>(1.0) / h;
  const Real invH2 = static_cast<Real>(1.0) / h2;
  Real density_error = 0.0;
  int fluid_particle_neighbor_count = 0;

  if(m_use_MLS_pressure)
  {
    if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
    {
      forall_boundary_model_Akinci12(
          bm_i->computeBoundaryPressureMLS();
        );
    }
  }

  START_TIMING("pressureGradientIter");
  computeRigidBodyGradient(fluidModelIndex, 0);
  STOP_TIMING_AVG("pressureGradientIter");

  START_TIMING("pressureSolverIter");
  #pragma omp parallel default(shared)
  {
    //////////////////////////////////////////////////////////////////////////
    // Compute pressure forces
    //////////////////////////////////////////////////////////////////////////
    //#pragma omp for reduction(+:fluid_particle_neighbor_count) schedule(static)
    #pragma omp for schedule(static)
    for (int i = 0; i < numParticles; i++)
    {
      if (model->getParticleState(i) != ParticleState::Active)
        continue;

      //////////////////////////////////////////////////////////////////////////
      // Evaluate rhs
      //////////////////////////////////////////////////////////////////////////
      const Real b_i = m_simulationData.getDensityAdv(fluidModelIndex, i) - static_cast<Real>(1.0);
      const Real ki = b_i*m_simulationData.getFactor(fluidModelIndex, i)*invH2;
      model->getPressure(i) = ki * model->getDensity(i);

      if(m_use_pressure_warmstart)
        m_simulationData.getKappa(fluidModelIndex, i) += ki;

      Vector3r &v_i = model->getVelocity(i);
      const Vector3r &xi = model->getPosition(i);

      //////////////////////////////////////////////////////////////////////////
      // Fluid
      //////////////////////////////////////////////////////////////////////////
      forall_fluid_neighbors(
        const Real b_j = m_simulationData.getDensityAdv(pid, neighborIndex) - static_cast<Real>(1.0);
        const Real kj = b_j*m_simulationData.getFactor(pid, neighborIndex)*invH2;
        const Real kSum = ki + fm_neighbor->getDensity0()/density0 * kj;
        if (fabs(kSum) > m_eps)
        {
        const Vector3r grad_p_j = -fm_neighbor->getVolume(neighborIndex) *sim->gradW(xi - xj);

        // Directly update velocities instead of storing pressure accelerations
        v_i -= h * kSum * grad_p_j;     // ki, kj already contain inverse density
      }
      )

      //////////////////////////////////////////////////////////////////////////
      // Boundary
      //////////////////////////////////////////////////////////////////////////
      if (fabs(ki) > m_eps)
      {
        if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
        {
          //forall_boundary_neighbors(
          for (unsigned int pid = nFluids; pid < sim->numberOfPointSets(); pid++)
            {
              BoundaryModel_Akinci2012 *bm_neighbor = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModelFromPointSet(pid));
              for (unsigned int j = 0; j < sim->numberOfNeighbors(fluidModelIndex, pid, i); j++)
                {
                  const unsigned int neighborIndex = sim->getNeighbor(fluidModelIndex, pid, i, j);
                  const Vector3r &xj = bm_neighbor->getPosition(neighborIndex);

                  const Vector3r grad_p_j = -bm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
                  // Directly update velocities instead of storing pressure accelerations
                  Vector3r velChange = Vector3r::Zero();// kj already contains inverse density
                  if(m_use_MLS_pressure)
                  {
                    velChange = -h * bm_neighbor->getPressure(neighborIndex) / (model->getDensity(i)) * grad_p_j; 
                  }
                  else
                  {
                    velChange = -h * (Real) 1.0 * ki * grad_p_j;       // kj already contains inverse density
                  }
                  
                  v_i += velChange;
                  Vector3r force =  -model->getMass(i) * velChange * invH;
                  //if(sim->isDebug())
                  //{
                  //if(j == 0 && bm_neighbor->getRigidBodyObject()->isDynamic())
                  //{
                  //LOG_INFO << RedHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] force = " << force.transpose() << CyanTail();
                  //}
                  //}
                  //if(force.norm() > 1e-6 && bm_neighbor->getRigidBodyObject()->isDynamic())
                    //fluid_particle_neighbor_count +=1;
                  bm_neighbor->addForce(xj,force);
                  // Note: This implementation is wrong !!!, still meet with write race hazard
                  //if(m_use_strong_coupling)
                  //{
                    //sim->getSimulatorBase()->getBoundarySimulator()->updateVelocity(); 
                    // TODO: add one step iteration of rigid contact solver here 
                    // sim->getSimulatorBase()->getBoundarySimulator()->getRigidContactSolver()->step(); 
                  //}
#ifdef BACKWARD
                  // checkout boundary is dynamic, if static, no need to calculate gradient
                  if(bm_neighbor->getRigidBodyObject()->isDynamic()
                    && false == bm_neighbor->getRigidBodyObject()->isAnimated())
                  {
                    total_force += force;
                    //this->computeGradient(bm_neighbor, fluidModelIndex, pid, i, neighborIndex, b_i, ki, force, 0);
                  }
#endif

                }
            }
        }
      }
    }
  }

    //////////////////////////////////////////////////////////////////////////
    // Update rho_adv and density error
    //////////////////////////////////////////////////////////////////////////
  #pragma omp parallel default(shared)
  {
    #pragma omp for reduction(+:density_error) schedule(static)
    for (int i = 0; i < numParticles; i++)
      {
        computeDensityAdv(fluidModelIndex, i, numParticles, h, density0);

        density_error += density0 * m_simulationData.getDensityAdv(fluidModelIndex, i) - density0;
      }

  }
  //if(sim->isDebug())
    //LOG_INFO << GreenHead() << " pressure solve : " << GreenTail();
  avg_density_err = density_error / numParticles;

  STOP_TIMING_AVG("pressureSolverIter");
  return fluid_particle_neighbor_count;
}

// ==========================================================================================
// Backward gradient
// ==========================================================================================
#ifdef BACKWARD
void TimeStepDiffDFSPH::computeRigidBodyGradient(const unsigned int fluidModelIndex, 
                                  const unsigned int mode )
{
  Simulation *sim = Simulation::getCurrent();
  const Real dt = TimeManager::getCurrent()->getTimeStepSize();
  const Real invH = static_cast<Real>(1.0)/ dt;
  const Real invH2 =static_cast<Real>(1.0)/ dt / dt;

  FluidModel *fluid_model = sim->getFluidModel(fluidModelIndex);

  for(int bm_index = 0; bm_index < sim->numberOfBoundaryModels(); bm_index++) 
  { 
    BoundaryModel_Akinci2012 *bm = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(bm_index)); 
    if(bm->getRigidBodyObject()->isDynamic() && bm->getRigidBodyObject()->isAnimated() == false)
    {
      #pragma omp parallel default(shared)
      {
        const unsigned numBoundaryParticles = bm->numberOfParticles(); 
        // Parallelly traverse all rigid body particles 
        #pragma omp for schedule(static)
        for (int j = 0; j < (int)numBoundaryParticles; j++)
        {
          const Vector3r xj = bm->getPosition(j);
          for (unsigned int pid = 0; pid < sim->numberOfFluidModels(); pid++)
          {
            auto fluid_neighbor = sim->getFluidModel(pid);
            auto n_fluid_neigbor_particle = sim->numberOfNeighbors(bm->getPointSetIndex(), pid, j);
            // Sequentially traverse all fluid neighbor particles
            for(unsigned int n = 0; n < n_fluid_neigbor_particle; n++)
            {
              // Note: n is not the real index of fluid neighbor, need to use getNeighbor 
              const unsigned int i = sim->getNeighbor(bm->getPointSetIndex(), pid, j, n);
              const Vector3r xi = fluid_model->getPosition(i);

              Real b_i = 0.0;
              if(mode == 0)
                b_i = m_simulationData.getDensityAdv(fluidModelIndex, i) - static_cast<Real>(1.0);
              else 
                b_i = m_simulationData.getDensityAdv(fluidModelIndex, i);
                
              unsigned int coeff = (mode == 0)? invH2 : invH; 
              const Real ki = b_i*m_simulationData.getFactor(fluidModelIndex, i)*coeff; 
              const Vector3r grad_p_j = -bm->getVolume(j) * sim->gradW(xi - xj);
              Vector3r velChange = -dt  * (Real) 1.0 * ki * grad_p_j;       // kj already contains inverse density
              Vector3r force =  -fluid_model->getMass(i) * velChange * invH;
              computeGradient(bm, fluidModelIndex, bm->getPointSetIndex(), i, j, b_i, ki, force, mode);  
            }
          }
        }
      }
    }
  }

}

void TimeStepDiffDFSPH::computeGradient( BoundaryModel_Akinci2012* bm_neighbor,
                                        const unsigned int fluidModelIndex,
                                        const unsigned int pid,
                                        const unsigned int fluidParticleIndex,
                                        const unsigned int rigidParticleIndex,
                                        const Real b_i,
                                        const Real ki,
                                        const Vector3r force,
                                        const unsigned int mode)
{
  const unsigned int i = fluidParticleIndex;
  const Real dt = TimeManager::getCurrent()->getTimeStepSize();
  const Real invH = static_cast<Real>(1.0)/ dt;
  const Real invH2 =static_cast<Real>(1.0)/ dt / dt;
  Simulation *sim = Simulation::getCurrent();
  FluidModel *model = sim->getFluidModel(fluidModelIndex);

  // Note: xj is the boundary particle
  const Vector3r &xi = model->getPosition(i);
  const Vector3r &xj = bm_neighbor->getPosition(rigidParticleIndex);
  const Vector3r grad_p_j = -bm_neighbor->getVolume(rigidParticleIndex) * sim->gradW(xi - xj);

  // Note: no need to add minus here since grad_p_j already has one
  Matrix3r grad2_p_j_to_xj =  bm_neighbor->getVolume(rigidParticleIndex) * sim->gradGradW(xi - xj);
  //LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_p_j_to_xj = " << grad_p_j_to_xj <<  YellowTail();

  auto factor = m_simulationData.getFactor(fluidModelIndex, i);

  Vector3r grad_b_i_to_xj;
  if(mode == 0)
    grad_b_i_to_xj = computeGradDensityAdvToBoundaryParticleX(fluidModelIndex, pid, i, rigidParticleIndex);
  else
    grad_b_i_to_xj = computeGradDensityChangeToBoundaryParticleX(fluidModelIndex, pid, i, rigidParticleIndex);

  //LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_b_i_to_xj = " << grad_b_i_to_xj.transpose()<< YellowTail();
  Vector3r grad_factor_to_xj = computeGradFactorToBoundaryParticleX(fluidModelIndex, pid, i, rigidParticleIndex);
  //LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_factor_to_xj = " << grad_factor_to_xj.transpose()<< YellowTail();
  // Note: const Real ki = b_i * m_simulationData.getFactor(fluidModelIndex, i);

  Real coeff = invH2;
  if (mode != 0)
    coeff = invH;
  Vector3r grad_ki_to_xj  = (grad_b_i_to_xj *  factor + b_i * grad_factor_to_xj) * coeff;
  //if(m_is_debug && j == 0)
  //LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_ki_to_xj = " << grad_ki_to_xj.transpose()<< YellowTail();

  // TODO: check this transpose here
  Matrix3r grad_velChange_to_xj = - (Real) 1. *
    (grad_p_j * grad_ki_to_xj.transpose() + ki * grad2_p_j_to_xj);
  if(m_is_debug)
  {
    LOG_INFO << RedHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_velChange_to_xj = \n" << grad_velChange_to_xj << YellowTail();

    LOG_INFO << GreenHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_p_j = \n" << grad_p_j.transpose() << CyanTail();
    LOG_INFO << GreenHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_ki_to_xj = \n" << grad_ki_to_xj.transpose()<< YellowTail();
    LOG_INFO << GreenHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] ki = " << ki <<YellowTail();
    LOG_INFO << GreenHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad2_p_j_to_xj = \n" << grad2_p_j_to_xj << YellowTail();
    LOG_INFO << GreenHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] W = \n" <<  sim->W(xi - xj) << YellowTail();
    LOG_INFO << GreenHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] gradW = \n" <<  sim->gradW(xi - xj) << YellowTail();
    LOG_INFO << GreenHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] gradGradW =  \n" <<  sim->gradGradW(xi - xj) << YellowTail() ;
    LOG_INFO << GreenHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] gradGradW_to_xj = \n" <<  -sim->gradGradW(xi - xj) << YellowTail() ;

    LOG_INFO << CyanHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] mass_i = \n" << model->getMass(i)<< YellowTail() ;
    LOG_INFO << CyanHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] V_bj = \n" <<  bm_neighbor->getVolume(rigidParticleIndex)  << YellowTail() ;

    LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_factor_to_xj = \n" << grad_factor_to_xj << YellowTail();
    LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_b_i_to_xj = \n" << grad_b_i_to_xj << YellowTail();
    LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] factor = \n" << factor << YellowTail();
    LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] b_i \n" << b_i << YellowTail() << "\n";

  }

  // ----------------- grad_force_to_v -----------------

  // TODO: do we need to consider grad v_i to vj ? To begin with we ignore this
  Vector3r grad_b_i_to_vj;
  if (mode == 0)
    grad_b_i_to_vj= computeGradDensityAdvToBoundaryParticleV(fluidModelIndex, pid, i, rigidParticleIndex);
  else
    grad_b_i_to_vj= computeGradDensityChangeToBoundaryParticleV(fluidModelIndex, pid, i, rigidParticleIndex);
  //LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_b_i_to_vj = " << grad_b_i_to_vj.transpose()<< YellowTail();
  // Note: pure derivative: grad_factor_to_vj = 0;
  Vector3r grad_ki_to_vj = grad_b_i_to_vj * factor * coeff;
  //LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_ki_to_vj = " << grad_ki_to_vj.transpose()<< YellowTail();

  // Note: pure derivative: grad_p_j_to_vj =  0;
  Matrix3r grad_velChange_to_vj = - (Real) 1. *
    (grad_p_j * grad_ki_to_vj.transpose());

  // -----------------------------------------------
  // Fluid gradient
  // -----------------------------------------------
  //Warning: need to consider all fluid neighbors fj of fi to compute grad_ki_to_vi?   
  // Such as considering fj in compute grad_b_i_to_vi; 
  // Furthermore, need to condiser how the perturbation of vi influence the physical properties of fj 
  Vector3r grad_b_i_to_vi;
  if(mode == 0)
    // Note: This computeGradDensityAdvToFluidParticleV is incomplete, we assume 
    // the physical propeties of all fj neighboring fi do not change 
    grad_b_i_to_vi = computeGradDensityAdvToFluidParticleV(fluidModelIndex, i);
  else
    grad_b_i_to_vi = computeGradDensityChangeToFluidParticleV(fluidModelIndex, i);
  Vector3r grad_ki_to_vi = grad_b_i_to_vi * factor * coeff; 
  // Note: grad_p_j_to_vi = 0 
  Matrix3r grad_velChange_to_vi = - (grad_p_j * grad_ki_to_vi.transpose());  
  Matrix3r grad_force_to_vi = - model->getMass(i) * grad_velChange_to_vi; 
  Matrix3r grad_vi_to_vj = grad_velChange_to_vj; 
  // -----------------------------------------------

  //if(m_is_debug&& j == 0){
  //LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_velChange_to_vj = " << grad_velChange_to_vj.transpose()<< YellowTail();
  //LOG_INFO << YellowHead() << " "<< YellowTail();
  //}

  // Note: no need to add invH here since we don't add h in grad_velChange
  Matrix3r grad_force_to_xj = -model->getMass(i) * grad_velChange_to_xj ; // pure derivative
  Matrix3r grad_force_to_vj = (Matrix3r::Identity() - dt * grad_velChange_to_vi).inverse() * -model->getMass(i) * grad_velChange_to_vj; // pure derivative
  
  //if(sim->isDebug())
  //{
    //LOG_INFO << RedHead() << "fluid vi gradient = \n" << grad_force_to_vi * grad_vi_to_vj << RedTail();
    //LOG_INFO << RedHead() << "rigid vj gradient = \n" << -model->getMass(i) * grad_velChange_to_vj  << RedTail();
    //LOG_INFO << RedHead() << " (Matrix3r::Identity() - dt * grad_velChange_to_vi) = \n" << (Matrix3r::Identity() - dt * grad_velChange_to_vi) << RedTail();
    //LOG_INFO << RedHead() << "inverse = \n" << (Matrix3r::Identity() - dt * grad_velChange_to_vi).inverse() << RedTail();
  //}

  //if(m_is_debug && j == 0)
  //{
  //LOG_INFO << CyanHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_force_to_xj = " <<grad_force_to_xj  << CyanTail();
  //LOG_INFO << CyanHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_force_to_vj = " <<grad_force_to_vj  << CyanTail();
  //}
  // gradVector3r v force to rigid body vj_n
  //bm_neighbor->get_grad_force_to_v(neighborIndex) += - model->getMass(i) * invH * (grad_velChange_to_vj + dt * grad_velChange_to_xj);
  bm_neighbor->get_grad_force_to_x(rigidParticleIndex) += grad_force_to_xj;
  if(m_is_debug)
  {
    LOG_INFO << GreenHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] invH = \n" << invH  << CyanTail();
    LOG_INFO << GreenHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_force_to_xj = \n" <<grad_force_to_xj  << CyanTail();
    //LOG_INFO << CyanHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_force_to_vj = " <<grad_force_to_vj  << CyanTail();
  }

  bm_neighbor->get_grad_force_to_v(rigidParticleIndex) += grad_force_to_vj;

  // ----------------- grad_torque_to_omega & v -----------------
  if(m_optimize_rotation)
  {
    // [0, rj] = q * [0, r0] * q.inverse()
    Vector3r rj = xj - bm_neighbor->getRigidBodyObject()->getPosition(); // Ref: SimulatorBase::updateBoundaryParticles
    // Wrong: auto r0 = bm_neighbor->getposition0(neighborindex) - m_simulationdata.get_init_x_rb();
    Vector3r r0 = bm_neighbor->getPosition0(rigidParticleIndex) ; // Ref: SimulatorBase::updateBoundaryParticles
    Quaternionr q = bm_neighbor->getRigidBodyObject()->getRotation();
    Vector3r qv = Vector3r(q.x(), q.y(), q.z());

    // Ref: Section3. Quaternions of "Position and Orientation Based Cosserat Rods"
    Matrix34r grad_rj_to_q = get_grad_Rqp_to_q(q, r0);
    //if(m_is_debug && j == 0)
    //LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_rj_to_q = " << grad_rj_to_q<< YellowTail();

    // xj = x_rb + rj = x_rb + R(q) r_j0
    Vector3r omega = bm_neighbor->getRigidBodyObject()->getAngularVelocity();
    Matrix34r grad_force_to_quaternion =  grad_force_to_xj * grad_rj_to_q + grad_force_to_vj * skewMatrix(omega) * grad_rj_to_q;
    //if(m_is_debug)
    //{
    //LOG_INFO << GreenHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_rj_to_q = \n" << grad_rj_to_q  << CyanTail();
    //LOG_INFO << GreenHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_force_to_quaternion = \n" << grad_force_to_quaternion  << CyanTail();
    //LOG_INFO << CyanHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_force_to_vj = " <<grad_force_to_vj  << CyanTail();
    //}
    //if(m_is_debug && j == 0)
    //LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_force_to_quaternion = " << grad_force_to_quaternion<< YellowTail();

    // vj = v_rb + omega \cross r_j = v_rb + [omega]rj = v_rb - [rj]omega
    Matrix3r grad_force_to_omega = grad_force_to_vj * skewMatrix(rj).transpose() ;
    //if(m_is_debug && j == 0)
    //LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_force_to_omega = " <<grad_force_to_omega  << YellowTail();

    // torque = rj \cross force = [rj]force = [force]^T rj
    Matrix34r grad_torque_to_quaternion = skewMatrix(rj) * grad_force_to_quaternion
      + skewMatrix(force).transpose() * grad_rj_to_q ;
    Matrix3r grad_torque_to_omega =  skewMatrix(rj) * grad_force_to_omega;
    //if(m_is_debug && j == 0)
    //{

    //LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_torque_to_quaternion = " <<grad_torque_to_quaternion  << YellowTail();
    //LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] grad_torque_to_omega = " << grad_torque_to_omega<< YellowTail();
    //}

    bm_neighbor->get_grad_force_to_omega(rigidParticleIndex) += grad_force_to_omega;
    bm_neighbor->get_grad_force_to_quaternion(rigidParticleIndex) += grad_force_to_quaternion;

    bm_neighbor->get_grad_torque_to_omega(rigidParticleIndex) += grad_torque_to_omega;
    bm_neighbor->get_grad_torque_to_quaternion(rigidParticleIndex) += grad_torque_to_quaternion;

    // ------------------------------------------------------------------------------------------
    // torque = rj \cross force = [rj]force = -[force]rj
    // rj = R(qn) r0
    // here grad_rj_to_v_rb = 0
    bm_neighbor->get_grad_torque_to_v(rigidParticleIndex) += skewMatrix(rj) * grad_force_to_vj;
    bm_neighbor->get_grad_torque_to_x(rigidParticleIndex) += skewMatrix(rj) * grad_force_to_xj;

  }

}


Vector3r TimeStepDiffDFSPH::computeGradFactorToBoundaryParticleX(const unsigned int fluidModelIndex, const unsigned int pointSetId, const unsigned int fluidParticleIndex, const unsigned int BoundaryParticleIndex)
{
  auto factor = m_simulationData.getFactor(fluidModelIndex, fluidParticleIndex);
  //LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] factor = " << factor << YellowTail();
  if(factor >= 0.)
  {
    return Vector3r::Zero();
  }

  Simulation *sim = Simulation::getCurrent();
  FluidModel *model = sim->getFluidModel(fluidModelIndex);

  // Note: get record stored in forward pass
  auto sum_grad_p_k = m_simulationData.get_sum_grad_p_k(fluidModelIndex, fluidParticleIndex);
  //LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] sum_grad_p_k = " << sum_grad_p_k.transpose()<< YellowTail();

  auto xi = model->getPosition(fluidParticleIndex);
  BoundaryModel_Akinci2012 *bm_neighbor = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModelFromPointSet(pointSetId));

  const Vector3r &xj = bm_neighbor->getPosition(BoundaryParticleIndex);
  auto vol = bm_neighbor->getVolume(BoundaryParticleIndex);
  const Vector3r grad_p_j = -vol * sim->gradW(xi - xj);
  Matrix3r grad_grap_p_j = vol * sim->gradGradW(xi - xj);
  auto s1 = grad_grap_p_j;
  //LOG_INFO << YellowHead() << "[ iter " << m_simulationData.get_opt_iter() <<" ] s1 = " << s1 << YellowTail();
  //auto s2 = grad_grap_p_j * grad_p_j;

  // Note: no need to add s2 here, since in computeDFSPHFactor, contribution of boundary is not counted in s2 .
  auto grad_factor_to_x = static_cast<Real>(2.) * factor * factor * ( s1 * sum_grad_p_k) ;
  return grad_factor_to_x;

}

Vector3r TimeStepDiffDFSPH::computeGradDensityAdvToBoundaryParticleX(const unsigned int fluidModelIndex,
                                                                     const unsigned int pointSetId,
                                                                     const unsigned int fluidParticleIndex, const unsigned int BoundaryParticleIndex)
{
  auto densityAdv = m_simulationData.getDensityAdv(fluidModelIndex, fluidParticleIndex);
  if(densityAdv <= 1.0)
    return Vector3r::Zero();
  else{
      Simulation *sim = Simulation::getCurrent();
      FluidModel *model = sim->getFluidModel(fluidModelIndex);

      auto xi = model->getPosition(fluidParticleIndex);
      auto vi = model->getVelocity(fluidParticleIndex);

      BoundaryModel_Akinci2012 *bm_neighbor = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModelFromPointSet(pointSetId));
      //BoundaryModel_Akinci2012 *bm = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(1));

      const Vector3r &xj = bm_neighbor->getPosition(BoundaryParticleIndex);
      auto vj = bm_neighbor->getVelocity(BoundaryParticleIndex);
      auto vol = bm_neighbor->getVolume(BoundaryParticleIndex);

      auto grad_density_to_x =  - vol * sim->gradW(xi - xj);

      const Real density0 = model->getDensity0();
      TimeManager *tm = TimeManager::getCurrent();
      const Real h = tm->getTimeStepSize();

      // Note: delta += bm_neighbor->getVolume(neighborIndex) * (vi - vj).dot(sim->gradW(xi - xj));
      Vector3r grad_densityAdv_to_x = grad_density_to_x / density0 - h * vol * sim->gradGradW(xi - xj) *(vi - vj);
      return grad_densityAdv_to_x;
    }
}


Vector3r TimeStepDiffDFSPH::computeGradDensityChangeToBoundaryParticleX(const unsigned int fluidModelIndex,
                                                                        const unsigned int pointSetId,
                                                                        const unsigned int fluidParticleIndex, const unsigned int BoundaryParticleIndex)
{
  auto densityAdv = m_simulationData.getDensityAdv(fluidModelIndex, fluidParticleIndex);
  if(densityAdv <= 0.0)
    return Vector3r::Zero();
  else{
      Simulation *sim = Simulation::getCurrent();
      FluidModel *model = sim->getFluidModel(fluidModelIndex);

      auto xi = model->getPosition(fluidParticleIndex);
      auto vi = model->getVelocity(fluidParticleIndex);

      BoundaryModel_Akinci2012 *bm_neighbor = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModelFromPointSet(pointSetId));

      const Vector3r &xj = bm_neighbor->getPosition(BoundaryParticleIndex);
      auto vj = bm_neighbor->getVelocity(BoundaryParticleIndex);

      auto vol = bm_neighbor->getVolume(BoundaryParticleIndex);

      Vector3r grad_densityChange_to_x = - vol * sim->gradGradW(xi - xj) * (vi - vj);
      return grad_densityChange_to_x;
    }

}

Vector3r TimeStepDiffDFSPH::computeGradDensityAdvToBoundaryParticleV(const unsigned int fluidModelIndex,
                                                                     const unsigned int pointSetId,
                                                                     const unsigned int fluidParticleIndex, const unsigned int BoundaryParticleIndex)
{
  auto densityAdv = m_simulationData.getDensityAdv(fluidModelIndex, fluidParticleIndex);
  if(densityAdv <= 1.0)
    return Vector3r::Zero();
  else{
      Simulation *sim = Simulation::getCurrent();
      FluidModel *model = sim->getFluidModel(fluidModelIndex);

      auto xi = model->getPosition(fluidParticleIndex);
      BoundaryModel_Akinci2012 *bm_neighbor = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModelFromPointSet(pointSetId));

      const Vector3r &xj = bm_neighbor->getPosition(BoundaryParticleIndex);

      auto vol = bm_neighbor->getVolume(BoundaryParticleIndex);

      TimeManager *tm = TimeManager::getCurrent();
      const Real h = tm->getTimeStepSize();

      Vector3r grad_densityAdv_to_v = - h * vol * sim->gradW(xi - xj);
      return grad_densityAdv_to_v;
    }

}

Vector3r TimeStepDiffDFSPH::computeGradDensityChangeToBoundaryParticleV(const unsigned int fluidModelIndex,
                                                                        const unsigned int pointSetId,
                                                                        const unsigned int fluidParticleIndex, const unsigned int BoundaryParticleIndex)
{
  auto densityAdv = m_simulationData.getDensityAdv(fluidModelIndex, fluidParticleIndex);
  if(densityAdv <= 0.0)
    return Vector3r::Zero();
  else{
      Simulation *sim = Simulation::getCurrent();
      FluidModel *model = sim->getFluidModel(fluidModelIndex);

      auto xi = model->getPosition(fluidParticleIndex);
      BoundaryModel_Akinci2012 *bm_neighbor = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModelFromPointSet(pointSetId));

      const Vector3r &xj = bm_neighbor->getPosition(BoundaryParticleIndex);

      auto vol = bm_neighbor->getVolume(BoundaryParticleIndex);

      Vector3r grad_densityChange_to_v = - vol * sim->gradW(xi - xj);
      return grad_densityChange_to_v;
    }

}
#endif


Vector3r TimeStepDiffDFSPH::computeGradDensityAdvToFluidParticleV(const unsigned int fluidModelIndex, 
                                                   const unsigned int fluidParticleIndex)
{
  auto densityAdv = m_simulationData.getDensityAdv(fluidModelIndex, fluidParticleIndex);
  if(densityAdv <= 1.0)
    return Vector3r::Zero();
  else{
      Vector3r grad_densityAdv_to_v = Vector3r::Zero();          

      Simulation *sim = Simulation::getCurrent();
      FluidModel *model = sim->getFluidModel(fluidModelIndex);
      const unsigned int nFluids = sim->numberOfFluidModels();
      
      TimeManager *tm = TimeManager::getCurrent();
      const Real h = tm->getTimeStepSize();
      
      const unsigned int i = fluidParticleIndex; 
      auto xi = model->getPosition(fluidParticleIndex);

      forall_fluid_neighbors(
          grad_densityAdv_to_v += fm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
        );
      if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
      {
        forall_boundary_neighbors(
          grad_densityAdv_to_v += bm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
        );
      }
      grad_densityAdv_to_v *= h; 
      return grad_densityAdv_to_v; 
  }
}

Vector3r TimeStepDiffDFSPH::computeGradDensityChangeToFluidParticleV(const unsigned int fluidModelIndex, 
                                                   const unsigned int fluidParticleIndex)
{
  auto densityAdv = m_simulationData.getDensityAdv(fluidModelIndex, fluidParticleIndex);
  if(densityAdv <= 0.0)
    return Vector3r::Zero();
  else{
      Vector3r grad_densityChange_to_v = Vector3r::Zero();          

      Simulation *sim = Simulation::getCurrent();
      FluidModel *model = sim->getFluidModel(fluidModelIndex);
      const unsigned int nFluids = sim->numberOfFluidModels();
      
      TimeManager *tm = TimeManager::getCurrent();
      
      const unsigned int i = fluidParticleIndex; 
      auto xi = model->getPosition(fluidParticleIndex);

      forall_fluid_neighbors(
          grad_densityChange_to_v +=  fm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
        );
      if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
      {
        forall_boundary_neighbors(
          grad_densityChange_to_v +=  bm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
        );
      }
      return grad_densityChange_to_v; 
  }
}

// =============================== divergence solve ===============================

void TimeStepDiffDFSPH::warmstartDivergenceSolve(const unsigned int fluidModelIndex)
{
	const Real h = TimeManager::getCurrent()->getTimeStepSize();
	const Real invH = static_cast<Real>(1.0) / h;
	Simulation *sim = Simulation::getCurrent();
	FluidModel *model = sim->getFluidModel(fluidModelIndex);
	const Real density0 = model->getDensity0();
	const int numParticles = (int)model->numActiveParticles();
	if (numParticles == 0)
		return;

	const unsigned int nFluids = sim->numberOfFluidModels();
	const unsigned int nBoundaries = sim->numberOfBoundaryModels();


	#pragma omp parallel default(shared)
	{
		//////////////////////////////////////////////////////////////////////////
		// Divide by h^2, the time step size has been removed in 
		// the last step to make the stiffness value independent 
		// of the time step size
		//////////////////////////////////////////////////////////////////////////
		#pragma omp for schedule(static)  
		for (int i = 0; i < numParticles; i++)
		{
			//m_simulationData.getKappaV(fluidModelIndex, i) = static_cast<Real>(0.5)*max(m_simulationData.getKappaV(fluidModelIndex, i)*invH, -static_cast<Real>(0.5) * density0*density0);
			computeDensityChange(fluidModelIndex, i, h);
			if (m_simulationData.getDensityAdv(fluidModelIndex, i) > 0.0)				
				//m_simulationData.getKappaV(fluidModelIndex, i) = static_cast<Real>(0.5) * max(m_simulationData.getKappaV(fluidModelIndex, i), static_cast<Real>(-0.5)) * invH;
        m_simulationData.getKappaV(fluidModelIndex, i) = static_cast<Real>(0.5) *
            max(m_simulationData.getKappaV(fluidModelIndex, i), static_cast<Real>(-0.5))
            * invH;
			else
				m_simulationData.getKappaV(fluidModelIndex, i) = 0.0;
		}

		#pragma omp for schedule(static)  
		for (int i = 0; i < (int)numParticles; i++)
		{
			if (model->getParticleState(i) != ParticleState::Active)
			{
				m_simulationData.getKappaV(fluidModelIndex, i) = 0.0;
				continue;
			}

			//if (m_simulationData.getDensityAdv(fluidModelIndex, i) > 0.0)
			{
				Vector3r &vel = model->getVelocity(i);
				const Real ki = m_simulationData.getKappaV(fluidModelIndex, i);
        model->getPressure(i) = ki * model->getDensity(i);
				const Vector3r &xi = model->getPosition(i);

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				forall_fluid_neighbors(
					const Real kj = m_simulationData.getKappaV(pid, neighborIndex);

					const Real kSum = (ki + fm_neighbor->getDensity0() / density0 * kj);
					if (fabs(kSum) > m_eps)
					{
						const Vector3r grad_p_j = -fm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
						vel -= h * kSum * grad_p_j;					// ki, kj already contain inverse density
					}
				)

				//////////////////////////////////////////////////////////////////////////
				// Boundary
				//////////////////////////////////////////////////////////////////////////
				if (fabs(ki) > m_eps)
				{
					if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
					{
						forall_boundary_neighbors(
							const Vector3r grad_p_j = -bm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
							const Vector3r velChange = -h * (Real) 1.0 * ki * grad_p_j;				// kj already contains inverse density
							vel += velChange;
              Vector3r force = -model->getMass(i) * velChange * invH;
							bm_neighbor->addForce(xj, force);
              if(bm_neighbor->getRigidBodyObject()->isDynamic()
                && false == bm_neighbor->getRigidBodyObject()->isAnimated())
              {
                total_force += force;
              }
						);
					}
				}
			}
		}
	}
}

void TimeStepDiffDFSPH::divergenceSolveIteration(const unsigned int fluidModelIndex, Real & avg_density_err)
{
  Simulation *sim = Simulation::getCurrent();
  FluidModel *model = sim->getFluidModel(fluidModelIndex);
  const Real density0 = model->getDensity0();
  const int numParticles = (int)model->numActiveParticles();
  if (numParticles == 0)
    return;

  const unsigned int nFluids = sim->numberOfFluidModels();
  const unsigned int nBoundaries = sim->numberOfBoundaryModels();
  const Real h = TimeManager::getCurrent()->getTimeStepSize();
  const Real invH = static_cast<Real>(1.0) / h;
  Real density_error = 0.0;

  if(m_use_MLS_pressure)
  {
    if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
    {
      forall_boundary_model_Akinci12(
          bm_i->computeBoundaryPressureMLS();
        );
    }
  }

  START_TIMING("divergenceGradientIter");
  computeRigidBodyGradient(fluidModelIndex, 1 );
  STOP_TIMING_AVG("divergenceGradientIter");
  //////////////////////////////////////////////////////////////////////////
  // Perform Jacobi iteration over all blocks
  //////////////////////////////////////////////////////////////////////////
  
  START_TIMING("divergenceSolverIter");
  #pragma omp parallel default(shared)
  {
    #pragma omp for schedule(static)
    for (int i = 0; i < (int)numParticles; i++)
      {
        if (model->getParticleState(i) != ParticleState::Active)
          continue;

        //////////////////////////////////////////////////////////////////////////
        // Evaluate rhs
        //////////////////////////////////////////////////////////////////////////
        const Real b_i = m_simulationData.getDensityAdv(fluidModelIndex, i);
        const Real ki = b_i*m_simulationData.getFactor(fluidModelIndex, i) * invH;
        model->getPressure(i) = ki * model->getDensity(i);
        if(m_use_divergence_warmstart)
          m_simulationData.getKappaV(fluidModelIndex, i) += ki;

        Vector3r &v_i = model->getVelocity(i);

        const Vector3r &xi = model->getPosition(i);

        //////////////////////////////////////////////////////////////////////////
        // Fluid
        //////////////////////////////////////////////////////////////////////////
        forall_fluid_neighbors(
          const Real b_j = m_simulationData.getDensityAdv(pid, neighborIndex);
          const Real kj = b_j*m_simulationData.getFactor(pid, neighborIndex) * invH;

          const Real kSum = ki + fm_neighbor->getDensity0() / density0 * kj;
          if (fabs(kSum) > m_eps)
          {
          const Vector3r grad_p_j = -fm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
          v_i -= h * kSum * grad_p_j;     // ki, kj already contain inverse density
        }
        )

        //////////////////////////////////////////////////////////////////////////
        // Boundary
        //////////////////////////////////////////////////////////////////////////
        if (fabs(ki) > m_eps)
        {
          if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
          {
            for (unsigned int pid = nFluids; pid < sim->numberOfPointSets(); pid++)
              {
                BoundaryModel_Akinci2012* bm_neighbor = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModelFromPointSet(pid));
                for (unsigned int j = 0; j < sim->numberOfNeighbors(fluidModelIndex, pid, i); j++)
                {
                  const unsigned int neighborIndex = sim->getNeighbor(fluidModelIndex, pid, i, j);
                  const Vector3r& xj = bm_neighbor->getPosition(neighborIndex);
                  const Vector3r grad_p_j = -bm_neighbor->getVolume(neighborIndex) * sim->gradW(xi - xj);
                  Vector3r velChange = Vector3r::Zero();// kj already contains inverse density
                  if(m_use_MLS_pressure)
                  {
                    velChange = -h * bm_neighbor->getPressure(neighborIndex) / (model->getDensity(i)) * grad_p_j; 
                  }
                  else
                  {
                    velChange = -h * (Real) 1.0 * ki * grad_p_j;       // kj already contains inverse density
                  }
                  v_i += velChange;
                  const Vector3r force = -model->getMass(i) * velChange * invH;
                  bm_neighbor->addForce(xj, force);
#ifdef BACKWARD
                  // checkout boundary is dynamic, if static, no need to calculate gradient
                  if(bm_neighbor->getRigidBodyObject()->isDynamic()
                    && false == bm_neighbor->getRigidBodyObject()->isAnimated())
                  {
                    //#pragma omp atomic
                    total_force += force; // FIXME: write race
                    //this->computeGradient(bm_neighbor, fluidModelIndex, pid, i, neighborIndex, b_i, ki, force, 1 );
                  }
#endif
                }
              }

          }

        }
      }
  }


  #pragma omp parallel default(shared)
  {
    // update rho_adv and density_error
    #pragma omp for reduction(+: density_error) schedule(static)
    for(int i = 0; i < (int)numParticles; i++)
      {
        computeDensityChange(fluidModelIndex, i, h);
        density_error += density0 * m_simulationData.getDensityAdv(fluidModelIndex, i);
      }
  }

  //if(sim->isDebug())
    //LOG_INFO << GreenHead() << " div solve : " << GreenTail();
  avg_density_err = density_error / numParticles;

  STOP_TIMING_AVG("divergenceSolverIter");
}

// =====================================================================

void TimeStepDiffDFSPH::computeDensityAdv(const unsigned int fluidModelIndex, const unsigned int i, const int numParticles, const Real h, const Real density0)
{
  Simulation *sim = Simulation::getCurrent();
  FluidModel *model = sim->getFluidModel(fluidModelIndex);
  const Real &density = model->getDensity(i);
  Real &densityAdv = m_simulationData.getDensityAdv(fluidModelIndex, i);
  const Vector3r &xi = model->getPosition(i);
  const Vector3r &vi = model->getVelocity(i);
  Real delta = 0.0;
  const unsigned int nFluids = sim->numberOfFluidModels();
  const unsigned int nBoundaries = sim->numberOfBoundaryModels();

  //////////////////////////////////////////////////////////////////////////
  // Fluid
  //////////////////////////////////////////////////////////////////////////
  forall_fluid_neighbors(
    const Vector3r &vj = fm_neighbor->getVelocity(neighborIndex);
    delta += fm_neighbor->getVolume(neighborIndex) * (vi - vj).dot(sim->gradW(xi - xj));
  )

  //////////////////////////////////////////////////////////////////////////
  // Boundary
  //////////////////////////////////////////////////////////////////////////
  if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
  {
    forall_boundary_neighbors(
      const Vector3r &vj = bm_neighbor->getVelocity(neighborIndex);
      delta += bm_neighbor->getVolume(neighborIndex) * (vi - vj).dot(sim->gradW(xi - xj));
    );
  }
  else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
  {
    forall_density_maps(
      Vector3r vj;
    bm_neighbor->getPointVelocity(xi, vj);
    delta -= (vi - vj).dot(gradRho);
  );
  }
  else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
  {
    forall_volume_maps(
      Vector3r vj;
    bm_neighbor->getPointVelocity(xj, vj);
    delta += Vj * (vi - vj).dot(sim->gradW(xi - xj));
  );
  }

  densityAdv = density / density0 + h*delta;
  densityAdv = max(densityAdv, static_cast<Real>(1.0));
}

void TimeStepDiffDFSPH::computeDensityChange(const unsigned int fluidModelIndex, const unsigned int i, const Real h)
{
  Simulation *sim = Simulation::getCurrent();
  FluidModel *model = sim->getFluidModel(fluidModelIndex);
  Real &densityAdv = m_simulationData.getDensityAdv(fluidModelIndex, i);
  const Vector3r &xi = model->getPosition(i);
  const Vector3r &vi = model->getVelocity(i);
  densityAdv = 0.0;
  unsigned int numNeighbors = 0;
  const unsigned int nFluids = sim->numberOfFluidModels();
  const unsigned int nBoundaries = sim->numberOfBoundaryModels();

  //////////////////////////////////////////////////////////////////////////
  // Fluid
  //////////////////////////////////////////////////////////////////////////
  forall_fluid_neighbors(
    const Vector3r &vj = fm_neighbor->getVelocity(neighborIndex);
    densityAdv += fm_neighbor->getVolume(neighborIndex) * (vi - vj).dot(sim->gradW(xi - xj));
  );

  //////////////////////////////////////////////////////////////////////////
  // Boundary
  //////////////////////////////////////////////////////////////////////////
  if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
  {
    forall_boundary_neighbors(
      const Vector3r &vj = bm_neighbor->getVelocity(neighborIndex);
      densityAdv += bm_neighbor->getVolume(neighborIndex) * (vi - vj).dot(sim->gradW(xi - xj));
    );
  }
else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
  {
    forall_density_maps(
      Vector3r vj;
    bm_neighbor->getPointVelocity(xi, vj);
    densityAdv -= (vi - vj).dot(gradRho);
  );
  }
else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
  {
    forall_volume_maps(
      Vector3r vj;
    bm_neighbor->getPointVelocity(xj, vj);
    densityAdv += Vj * (vi - vj).dot(sim->gradW(xi - xj));
  );
  }

  // only correct positive divergence
  densityAdv = max(densityAdv, static_cast<Real>(0.0));

  for (unsigned int pid = 0; pid < sim->numberOfPointSets(); pid++)
    numNeighbors += sim->numberOfNeighbors(fluidModelIndex, pid, i);

  // in case of particle deficiency do not perform a divergence solve
  if (!sim->is2DSimulation())
  {
    if (numNeighbors < 20)
      densityAdv = 0.0;
  }
else
  {
    if (numNeighbors < 7)
      densityAdv = 0.0;
  }
}

// ===========================================
void TimeStepDiffDFSPH::performNeighborhoodSearch()
{
  if (Simulation::getCurrent()->zSortEnabled())
  {
    if (m_counter % 500 == 0)
    {
      Simulation::getCurrent()->performNeighborhoodSearchSort();
      m_simulationData.performNeighborhoodSearchSort();
    }
    m_counter++;
  }
  Simulation::getCurrent()->performNeighborhoodSearch();
}

void TimeStepDiffDFSPH::reset()
{
  TimeStep::reset();
  m_simulationData.reset();
  m_counter = 0;
  step_count = 0;
  m_iterations = 0;
  m_iterationsV = 0;
}
void TimeStepDiffDFSPH::emittedParticles(FluidModel *model, const unsigned int startIndex)
{
  m_simulationData.emittedParticles(model, startIndex);
}
void TimeStepDiffDFSPH::resize()
{
  m_simulationData.init();
}

#ifdef BACKWARD
BoundaryModel_Akinci2012* TimeStepDiffDFSPH::get_boundary_model_Akinci12(unsigned int bm_index)
{
  assert(bm_index < Simulation::getCurrent()->numberOfBoundaryModels());
  BoundaryModel_Akinci2012 *bm = static_cast<BoundaryModel_Akinci2012 *>(Simulation::getCurrent()->getBoundaryModel(bm_index));
  assert(bm);
  return bm;
}

//------------------------------------------------------------------------------------------------

void TimeStepDiffDFSPH::set_init_v_rb(unsigned int bm_index, const Vector3r param)
{
  m_simulationData.get_init_v_rb(bm_index) = param;
}

void TimeStepDiffDFSPH::set_init_omega_rb(unsigned int bm_index, const Vector3r param)
{
  m_simulationData.get_init_omega_rb(bm_index) = param;
}

void TimeStepDiffDFSPH::set_init_omega_rb_to_joint(unsigned int bm_index, const Vector3r param)
{
  m_simulationData.get_init_omega_rb(bm_index) = param;
  Simulation *sim = Simulation::getCurrent();
  auto bm = sim->getBoundaryModel(bm_index);
  auto system = bm->getRigidBodyObject()->getArticulatedSystem();
  if(system)
  {
    RigidBodyObject *otherBody = nullptr;
    Vector3r l1, l2;
    system->getJointInfoByBody(bm->getRigidBodyObject(), otherBody, l1, l2);
    if(otherBody)
    {
      Vector3r RL1 = bm->getRigidBodyObject()->getRotation0().toRotationMatrix() * l1;
      Vector3r RL2 = otherBody->getRotation0().toRotationMatrix() * l2;
      m_simulationData.get_init_v_rb(bm_index) = otherBody->getVelocity0() + otherBody->getAngularVelocity0().cross(RL2) - param.cross(RL1);
    }
  }
}

Vector3r TimeStepDiffDFSPH::get_init_v_rb(unsigned int index)
{
  return m_simulationData.get_init_v_rb(index);
}
Vector3r TimeStepDiffDFSPH::get_init_omega_rb(unsigned int index)
{
  return m_simulationData.get_init_omega_rb(index);
}

//Vector3r TimeStepDiffDFSPH::get_final_angle_rb_in_radian()
//{
//return m_simulationData.get_final_angle_rb_in_radian();
//}
//Vector3r TimeStepDiffDFSPH::get_final_angle_rb_in_degree()
//{
//return m_simulationData.get_final_angle_rb_in_radian() / M_PI * 180.;
//}
//Vector4r TimeStepDiffDFSPH::get_final_quaternion_rb()
//{
//auto q = m_simulationData.get_final_quaternion_rb();
//return Vector4r(q.w(), q.x(), q.y(), q.z());
//}

Vector3r TimeStepDiffDFSPH::get_target_angle_in_radian(unsigned int index)
{
  return m_simulationData.get_target_angle_in_radian(index);
}
Vector3r TimeStepDiffDFSPH::get_target_x(unsigned int index)
{
  return m_simulationData.get_target_x(index);
}
Vector3r TimeStepDiffDFSPH::get_init_x(unsigned int index)
{
  return m_simulationData.get_init_x(index);
}
void TimeStepDiffDFSPH::set_target_x(unsigned int index, const Vector3r new_target_x)
{
  m_simulationData.get_target_x(index) = new_target_x;
}

Vector4r TimeStepDiffDFSPH::get_target_quaternion_vec4(unsigned int index)
{
  auto q = m_simulationData.get_target_quaternion(index);
  return Vector4r(q.w(), q.x(), q.y(), q.z());
}

Real TimeStepDiffDFSPH::get_loss()
{
  return m_simulationData.get_loss();
}
void TimeStepDiffDFSPH::set_loss(const Real new_loss)
{
  m_simulationData.get_loss() = new_loss;
}

Real TimeStepDiffDFSPH::get_loss_x()
{
  return m_simulationData.get_loss_x();
}
void TimeStepDiffDFSPH::set_loss_x(const Real new_loss)
{
  m_simulationData.get_loss_x() = new_loss;
}

Real TimeStepDiffDFSPH::get_loss_rotation()
{
  return m_simulationData.get_loss_rotation();
}
void TimeStepDiffDFSPH::set_loss_rotation(const Real new_loss)
{
  m_simulationData.get_loss_rotation() = new_loss;
}

Real TimeStepDiffDFSPH::get_lr()
{
  return m_simulationData.get_learning_rate();
}
void TimeStepDiffDFSPH::set_lr(const Real new_lr)
{
  m_simulationData.get_learning_rate() = new_lr;
}

bool TimeStepDiffDFSPH::is_trajectory_finish_callback()
{
  return m_is_trajectory_finished_callback;
}

void TimeStepDiffDFSPH::clear_all_callbacks()
{
  m_is_trajectory_finished_callback = false;
}

bool TimeStepDiffDFSPH::is_in_new_trajectory()
{
  return m_is_in_new_trajectory;
}

void TimeStepDiffDFSPH::set_in_new_trajectory(bool b)
{
  m_is_in_new_trajectory = b;
}

void TimeStepDiffDFSPH::set_custom_log_message(const std::string & s)
{
  m_simulationData.get_custom_log_message()+=s ;
}

string TimeStepDiffDFSPH::get_custom_log_message()
{
  return m_simulationData.get_custom_log_message();
}

void TimeStepDiffDFSPH::add_log(const std::string & s)
{
  LOG_INFO << s;
}

void TimeStepDiffDFSPH::reset_gradient()
{
  Simulation *sim = Simulation::getCurrent();
  forall_dynamic_boundary_model_Akinci12(
      bm_i->reset_gradient(); 
    );
}

unsigned int TimeStepDiffDFSPH::get_num_1ring_fluid_particle()
{
  return m_simulationData.num_1ring_fluid_particle; 
}

// ----------------------------------------------------------------------------------

void TimeStepDiffDFSPH::setlbfgsMaxIter(unsigned int m)
{
  lbfgs_wrapper.lbfgs_param.max_iterations = m;
}
void TimeStepDiffDFSPH::setlbfgsMemorySize(unsigned int m)
{
  lbfgs_wrapper.lbfgs_param.m = m;
}
void TimeStepDiffDFSPH::setlbfgsLineSearchMethod(unsigned int m)
{
  lbfgs_wrapper.lbfgs_param.linesearch = m;
}
void TimeStepDiffDFSPH::setlbfgsMaxLineSearch(unsigned int m)
{
  lbfgs_wrapper.lbfgs_param.max_linesearch = m;
}

void TimeStepDiffDFSPH::startlbfgsTraining()
{
  Real fx;
  try
  {
    int nIter = lbfgs_wrapper.lbfgs_solver.minimize(lbfgs_wrapper, lbfgs_wrapper.initPoint, fx);

    LOG_INFO << CyanHead() << "LBFGS training finish" << CyanTail() << '\n';
    LOG_INFO << CyanHead() << "n_iter = " << nIter << "\tloss = " << fx << CyanTail() << '\n';
    LOG_INFO << CyanHead() << "final_grad = " << lbfgs_wrapper.lbfgs_solver.final_grad().transpose() << CyanTail() << '\n';
  }
  catch(const std::exception &error)
  {
    LOG_INFO << RedHead() << "LBFGS terminated early: " << error.what() << RedTail() << '\n';
  }
}

void TimeStepDiffDFSPH::setlbfgsInitPoint(VectorXr& initPoint)
{
  LOG_INFO << GreenHead() << "LBFGS: set init point = " << initPoint.transpose() << GreenTail() << '\n';
  lbfgs_wrapper.initPoint = initPoint;
}

void TimeStepDiffDFSPH::setlbfgsUseNormalizedGrad(bool b)
{
  lbfgs_wrapper.useNormalizedGrad = b;
}

Real TimeStepDiffDFSPH::LBFGSFuncWrapper::operator()(const VectorXr &x, VectorXr &grad)
{
  VectorXr fx_grad = m_func(x);
  const int size = x.size();
  assert(fx_grad.size() == size + 1);
  assert(grad.size() == size);
  for (unsigned int i = 0; i < size; i++)
  {
    grad(i) = fx_grad(i);
  }
  LOG_INFO << CyanHead() << "LBFGS training x = " << x.transpose() << CyanTail() << '\n';
  LOG_INFO << CyanHead() << "LBFGS training grad = " << grad.transpose() << CyanTail() << '\n';
  if(useNormalizedGrad)
  {
    grad.normalize();
    LOG_INFO << CyanHead() << "LBFGS training normalized grad = " << grad.transpose() << CyanTail() << '\n';
  }
  LOG_INFO << CyanHead() << "LBFGS training loss = " << fx_grad(size) << CyanTail() << '\n';
  return fx_grad(size);
}

#endif
