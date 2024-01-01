#include "SimulationDataDiffDFSPH.h"
#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/DFSPH/SimulationDataDFSPH.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "SPlisHSPlasH/Simulation.h"
#include "Simulator/SceneConfiguration.h"
#include <cmath>

using namespace SPH; 

SimulationDataDiffDFSPH::SimulationDataDiffDFSPH():
	m_factor(),
	m_kappa(),
	m_kappaV(),
	m_density_adv()
#ifdef BACKWARD
  ,m_sum_grad_p_k_array(),
  target_time(SceneConfiguration::getCurrent()->getScene().targetTime),
  uniform_accelerate_rb_time_at_beginning(SceneConfiguration::getCurrent()->getScene().uniformAccelerateRBTime), // TODO: add SceneLoader
  n_boundary_models(SceneConfiguration::getCurrent()->getScene().boundaryModels.size()),

  loss(0.0),
  loss_x(0.0),
  loss_rotation(0.0),
  learning_rate(1.),
  m_opt_iter(0),

  total_mass_fluid(0.),
  total_mass_rb(0.),

  custom_log_msg(""), 

  num_1ring_fluid_particle(0)
#endif 
{
  target_x.reserve(n_boundary_models);
  init_x.reserve(n_boundary_models);
  target_quaternion.reserve(n_boundary_models);
  target_angle_in_radian.reserve(n_boundary_models);
  target_angle_in_degree.reserve(n_boundary_models);
  
  init_v_rb.reserve(n_boundary_models);
  init_omega_rb.reserve(n_boundary_models);

  init_rb_rotation.reserve(n_boundary_models);
  current_omega_rb.reserve(n_boundary_models);
  current_x_rb.reserve(n_boundary_models);
  current_v_rb.reserve(n_boundary_models);
  for(int i = 0; i < n_boundary_models; i++)
  {
    target_x[i] = SceneConfiguration::getCurrent()->getScene().boundaryModels[i]->target_x;
    init_x[i] = SceneConfiguration::getCurrent()->getScene().boundaryModels[i]->translation;
    target_angle_in_degree[i] = SceneConfiguration::getCurrent()->getScene().boundaryModels[i]->target_angle_in_degree;
    target_angle_in_radian[i] = target_angle_in_degree[i] / 180. * M_PI;
    get_target_quaternion(i);
    init_v_rb[i] = SceneConfiguration::getCurrent()->getScene().boundaryModels[i]->init_velocity;
    init_omega_rb[i] = SceneConfiguration::getCurrent()->getScene().boundaryModels[i]->init_angular_velocity;

    current_omega_rb[i].setZero(); 
    current_v_rb[i].setZero();
    current_x_rb[i].setZero();
  }
}

SimulationDataDiffDFSPH::~SimulationDataDiffDFSPH(void)
{
    cleanup();
}

void SimulationDataDiffDFSPH::init()
{
  Simulation *sim = Simulation::getCurrent();
	const unsigned int nModels = sim->numberOfFluidModels();

	m_factor.resize(nModels);
	m_kappa.resize(nModels);
	m_kappaV.resize(nModels);
	m_density_adv.resize(nModels);
#ifdef BACKWARD
  m_sum_grad_p_k_array.resize(nModels);
#endif 
	for (unsigned int i = 0; i < nModels; i++)
	{
		FluidModel *fm = sim->getFluidModel(i);
		m_factor[i].resize(fm->numParticles(), 0.0);
		m_kappa[i].resize(fm->numParticles(), 0.0);
		m_kappaV[i].resize(fm->numParticles(), 0.0);
		m_density_adv[i].resize(fm->numParticles(), 0.0);
#ifdef BACKWARD
		m_sum_grad_p_k_array[i].resize(fm->numParticles(), Vector3r::Zero());
#endif 
  }
}


void SimulationDataDiffDFSPH::cleanup()
{
 Simulation *sim = Simulation::getCurrent();
	const unsigned int nModels = sim->numberOfFluidModels();

	for (unsigned int i = 0; i < nModels; i++)
	{
		m_factor[i].clear();
		m_kappa[i].clear();
		m_kappaV[i].clear();
		m_density_adv[i].clear();
#ifdef BACKWARD
		m_sum_grad_p_k_array[i].clear();
#endif 
	}

	m_factor.clear();
	m_kappa.clear();
	m_kappaV.clear();
	m_density_adv.clear();
#ifdef BACKWARD
	m_sum_grad_p_k_array.clear();
#endif 

}

void SimulationDataDiffDFSPH::reset()
{
//#ifdef BACKWARD
 //loss = 0. ; // reset loss 
//#endif 

 Simulation *sim = Simulation::getCurrent();
	const unsigned int nModels = sim->numberOfFluidModels();

	for (unsigned int i = 0; i < nModels; i++)
	{
		FluidModel *fm = sim->getFluidModel(i);
		for (unsigned int j = 0; j < fm->numActiveParticles(); j++)
		{
			m_kappa[i][j] = 0.0;
			m_kappaV[i][j] = 0.0;

#ifdef BACKWARD
			m_sum_grad_p_k_array[i][j].setZero();

      //custom_log_msg = "";
#endif 
		}
	}

}


void SimulationDataDiffDFSPH::performNeighborhoodSearchSort()
{
	Simulation *sim = Simulation::getCurrent();
	const unsigned int nModels = sim->numberOfFluidModels();

	for (unsigned int i = 0; i < nModels; i++)
	{
		FluidModel *fm = sim->getFluidModel(i);
		const unsigned int numPart = fm->numActiveParticles();
		if (numPart != 0)
		{
			auto const& d = sim->getNeighborhoodSearch()->point_set(fm->getPointSetIndex());
			//d.sort_field(&m_factor[i][0]);
			d.sort_field(&m_kappa[i][0]);
			d.sort_field(&m_kappaV[i][0]);
			//d.sort_field(&m_density_adv[i][0]);
#ifdef BACKWARD
			d.sort_field(&m_sum_grad_p_k_array[i][0]);
#endif 
		}
	}
}

void SimulationDataDiffDFSPH::emittedParticles(FluidModel *model, const unsigned int startIndex)
{
	// initialize kappa values for new particles
	const unsigned int fluidModelIndex = model->getPointSetIndex();
	for (unsigned int j = startIndex; j < model->numActiveParticles(); j++)
	{
		m_kappa[fluidModelIndex][j] = 0.0;
		m_kappaV[fluidModelIndex][j] = 0.0;
	}
}

