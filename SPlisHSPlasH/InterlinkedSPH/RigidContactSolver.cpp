#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/Simulation.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "Utilities/ColorfulPrint.h"
#include "RigidContactSolver.h"
#include "SPlisHSPlasH/GradientUtils.h"
#include "Utilities/ColorfulPrint.h"
#include "Utilities/Logger.h"
#include "Utilities/Timing.h"
#include <ctime>
#include <chrono>
#include <utility>
#include <vector>

#define print(x) std::cout << Utilities::YellowHead() << #x << " = " << x << Utilities::YellowTail() << std::endl;

using namespace std;
using namespace SPH;

// int RigidContactSolver::FRICTION_COEFF = -1;

RigidContactSolver::RigidContactSolver()
{
  Simulation *sim = Simulation::getCurrent();
  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();

  use_penalty_force = true;

  use_friction = (sim->getFrictionCoeff() > 1e-5);
  friction_coeff = 0.0;
  num_contacts = 0;
  density_error_thresh = sim->getRRContactDensityErrorThresh(); // 1 %
  supportRadius = sim->getRRContactSupportRadiusFactor() * sim->getParticleRadius();

  vel_rr_R.resize(n_rigid_obj);
  omega_rr_R.resize(n_rigid_obj);

  n_total_rigid_particles = 0;

  pressure.resize(n_rigid_obj);
  nabla_pressure.resize(n_rigid_obj);
  density.resize(n_rigid_obj);
  density0.resize(n_rigid_obj);
  vel.resize(n_rigid_obj);
  vel_rr.resize(n_rigid_obj);
  vol.resize(n_rigid_obj);
  vol0.resize(n_rigid_obj);
  s.resize(n_rigid_obj);
  b_from_v_rr_r.resize(n_rigid_obj);
  b.resize(n_rigid_obj);
  div_vel_rr.resize(n_rigid_obj);
  div_v_s.resize(n_rigid_obj);

  // ----------------------------------------------------------

  grad_pressure_to_vel_before_contact.resize(n_rigid_obj);
  grad_pressure_to_omega_before_contact.resize(n_rigid_obj);
  grad_pressure_to_x_before_contact.resize(n_rigid_obj);
  grad_pressure_to_q_before_contact.resize(n_rigid_obj);

  grad_s_to_vel_before_contact.resize(n_rigid_obj);
  grad_s_to_omega_before_contact.resize(n_rigid_obj);
  grad_s_to_x_before_contact.resize(n_rigid_obj);
  grad_s_to_q_before_contact.resize(n_rigid_obj);

  grad_div_vel_rr_to_vel_before_contact.resize(n_rigid_obj);
  grad_div_vel_rr_to_omega_before_contact.resize(n_rigid_obj);
  grad_div_vel_rr_to_x_before_contact.resize(n_rigid_obj);
  grad_div_vel_rr_to_q_before_contact.resize(n_rigid_obj);

  grad_nabla_p_r_to_vel_before_contact.resize(n_rigid_obj);
  grad_nabla_p_r_to_omega_before_contact.resize(n_rigid_obj);
  grad_nabla_p_r_to_x_before_contact.resize(n_rigid_obj);
  grad_nabla_p_r_to_q_before_contact.resize(n_rigid_obj);

  grad_vel_rr_to_vel_before_contact.resize(n_rigid_obj);
  grad_vel_rr_to_omega_before_contact.resize(n_rigid_obj);
  grad_vel_rr_to_x_before_contact.resize(n_rigid_obj);
  grad_vel_rr_to_q_before_contact.resize(n_rigid_obj);

  grad_vel_rr_R_to_vel_before_contact.resize(n_rigid_obj);
  grad_vel_rr_R_to_omega_before_contact.resize(n_rigid_obj);
  grad_vel_rr_R_to_x_before_contact.resize(n_rigid_obj);
  grad_vel_rr_R_to_q_before_contact.resize(n_rigid_obj);

  grad_omega_rr_R_to_vel_before_contact.resize(n_rigid_obj);
  grad_omega_rr_R_to_omega_before_contact.resize(n_rigid_obj);
  grad_omega_rr_R_to_x_before_contact.resize(n_rigid_obj);
  grad_omega_rr_R_to_q_before_contact.resize(n_rigid_obj);

  grad_friction_to_vel_before_contact.resize(n_rigid_obj);
  grad_friction_to_omega_before_contact.resize(n_rigid_obj);
  grad_friction_to_x_before_contact.resize(n_rigid_obj);
  grad_friction_to_q_before_contact.resize(n_rigid_obj);

  grad_net_force_to_vel_before_contact.resize(n_rigid_obj);
  grad_net_force_to_omega_before_contact.resize(n_rigid_obj);
  grad_net_force_to_x_before_contact.resize(n_rigid_obj);
  grad_net_force_to_q_before_contact.resize(n_rigid_obj);

  grad_net_torque_to_vel_before_contact.resize(n_rigid_obj);
  grad_net_torque_to_omega_before_contact.resize(n_rigid_obj);
  grad_net_torque_to_x_before_contact.resize(n_rigid_obj);
  grad_net_torque_to_q_before_contact.resize(n_rigid_obj);

  for (int R = 0; R < n_rigid_obj; R++)
  {
    auto bm = static_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModel(R));
    unsigned n_particles = bm->numberOfParticles();
    n_total_rigid_particles += n_particles;

    pressure[R].resize(n_particles);
    nabla_pressure[R].resize(n_particles);
    density[R].resize(n_particles);
    density0[R].resize(n_particles);
    vel[R].resize(n_particles);
    vel_rr[R].resize(n_particles);
    vol[R].resize(n_particles);
    vol0[R].resize(n_particles);
    s[R].resize(n_particles);
    b_from_v_rr_r[R].resize(n_particles);
    b[R].resize(n_particles);
    div_vel_rr[R].resize(n_particles);
    div_v_s[R].resize(n_particles);

    vel_rr_R[R] = Vector3r::Zero();
    omega_rr_R[R] = Vector3r::Zero();

    // ----------------------------------------------------------

    grad_pressure_to_vel_before_contact[R].resize(n_particles);
    grad_pressure_to_omega_before_contact[R].resize(n_particles);
    grad_pressure_to_x_before_contact[R].resize(n_particles);
    grad_pressure_to_q_before_contact[R].resize(n_particles);
    
    grad_s_to_vel_before_contact[R].resize(n_particles);
    grad_s_to_omega_before_contact[R].resize(n_particles);
    grad_s_to_x_before_contact[R].resize(n_particles);
    grad_s_to_q_before_contact[R].resize(n_particles);

    grad_div_vel_rr_to_vel_before_contact[R].resize(n_particles);
    grad_div_vel_rr_to_omega_before_contact[R].resize(n_particles);
    grad_div_vel_rr_to_x_before_contact[R].resize(n_particles);
    grad_div_vel_rr_to_q_before_contact[R].resize(n_particles);

    grad_nabla_p_r_to_vel_before_contact[R].resize(n_particles);
    grad_nabla_p_r_to_omega_before_contact[R].resize(n_particles);
    grad_nabla_p_r_to_x_before_contact[R].resize(n_particles);
    grad_nabla_p_r_to_q_before_contact[R].resize(n_particles);

    grad_vel_rr_to_vel_before_contact[R].resize(n_particles);
    grad_vel_rr_to_omega_before_contact[R].resize(n_particles);
    grad_vel_rr_to_x_before_contact[R].resize(n_particles);
    grad_vel_rr_to_q_before_contact[R].resize(n_particles);

    // -----------------------------------------------------------------------------------
    grad_vel_rr_R_to_vel_before_contact[R].resize(n_rigid_obj);
    grad_vel_rr_R_to_omega_before_contact[R].resize(n_rigid_obj);
    grad_vel_rr_R_to_x_before_contact[R].resize(n_rigid_obj);
    grad_vel_rr_R_to_q_before_contact[R].resize(n_rigid_obj);

    grad_omega_rr_R_to_vel_before_contact[R].resize(n_rigid_obj);
    grad_omega_rr_R_to_omega_before_contact[R].resize(n_rigid_obj);
    grad_omega_rr_R_to_x_before_contact[R].resize(n_rigid_obj);
    grad_omega_rr_R_to_q_before_contact[R].resize(n_rigid_obj);

    grad_friction_to_vel_before_contact[R].resize(n_rigid_obj);
    grad_friction_to_omega_before_contact[R].resize(n_rigid_obj);
    grad_friction_to_x_before_contact[R].resize(n_rigid_obj);
    grad_friction_to_q_before_contact[R].resize(n_rigid_obj);

    grad_net_force_to_vel_before_contact[R].resize(n_rigid_obj);
    grad_net_force_to_omega_before_contact[R].resize(n_rigid_obj);
    grad_net_force_to_x_before_contact[R].resize(n_rigid_obj);
    grad_net_force_to_q_before_contact[R].resize(n_rigid_obj);

    grad_net_torque_to_vel_before_contact[R].resize(n_rigid_obj);
    grad_net_torque_to_omega_before_contact[R].resize(n_rigid_obj);
    grad_net_torque_to_x_before_contact[R].resize(n_rigid_obj);
    grad_net_torque_to_q_before_contact[R].resize(n_rigid_obj);
  }

  m_particle_indices.resize(n_total_rigid_particles);
  unsigned particle_counter = 0;

  for (int R = 0; R < n_rigid_obj; R++)
  {
    auto bm = static_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModel(R));
    unsigned n_particles = bm->numberOfParticles();

    for (int r = 0; r < n_particles; r++)
    {
      grad_pressure_to_vel_before_contact[R][r].resize(n_rigid_obj);
      grad_pressure_to_omega_before_contact[R][r].resize(n_rigid_obj);
      grad_pressure_to_x_before_contact[R][r].resize(n_rigid_obj);
      grad_pressure_to_q_before_contact[R][r].resize(n_rigid_obj);

      grad_s_to_vel_before_contact[R][r].resize(n_rigid_obj);
      grad_s_to_omega_before_contact[R][r].resize(n_rigid_obj);
      grad_s_to_x_before_contact[R][r].resize(n_rigid_obj);
      grad_s_to_q_before_contact[R][r].resize(n_rigid_obj);

      grad_div_vel_rr_to_vel_before_contact[R][r].resize(n_rigid_obj);
      grad_div_vel_rr_to_omega_before_contact[R][r].resize(n_rigid_obj);
      grad_div_vel_rr_to_x_before_contact[R][r].resize(n_rigid_obj);
      grad_div_vel_rr_to_q_before_contact[R][r].resize(n_rigid_obj);

      grad_nabla_p_r_to_vel_before_contact[R][r].resize(n_rigid_obj);
      grad_nabla_p_r_to_omega_before_contact[R][r].resize(n_rigid_obj);
      grad_nabla_p_r_to_x_before_contact[R][r].resize(n_rigid_obj);
      grad_nabla_p_r_to_q_before_contact[R][r].resize(n_rigid_obj);

      grad_vel_rr_to_vel_before_contact[R][r].resize(n_rigid_obj);
      grad_vel_rr_to_omega_before_contact[R][r].resize(n_rigid_obj);
      grad_vel_rr_to_x_before_contact[R][r].resize(n_rigid_obj);
      grad_vel_rr_to_q_before_contact[R][r].resize(n_rigid_obj);

      m_particle_indices[particle_counter] = std::pair<unsigned, unsigned>(R, r);
      particle_counter++;
    }
  }

  NeighborhoodSearch *neighborhoodSearch = Simulation::getCurrent()->getNeighborhoodSearch();
  neighborhoodSearch->set_active(true);
  neighborhoodSearch->find_neighbors();

  // Initialize the volume0 and density0 of each rigid particle
  Real gamma = sim->getRRContactGammaFactor();
  if (sim->is2DSimulation())
    gamma /= sim->getParticleRadius(); // What if in 3d we also divide with particle radius?

//#pragma omp parallel
  {
//#pragma omp for
    forall_rigid_particles(
      pressure[R][r] = 0.;
      Real sum_W = this->W_zero();
      forall_rigid_neighbors(
        if (R_k == R)
        {
          sum_W += this->W(x_r - x_k);
        }
      )
      vol0[R][r] = gamma / sum_W;
    );

  //#pragma omp for
    forall_rigid_particles(
      density0[R][r] = vol0[R][r] * this->W_zero();
      forall_rigid_neighbors(
        if (R_k == R)
        {
          density0[R][r] += vol0[R_k][k] * this->W(x_r - x_k); // based on SPH rules
        }
      )
    )
         
  }

  reset();
  performNeighborhoodSearchSort();
}

void RigidContactSolver::beforeIterationInitialize()
{
  Simulation *sim = Simulation::getCurrent();
  density_error_thresh = sim->getRRContactDensityErrorThresh(); // 1 %
  //
  NeighborhoodSearch *neighborhoodSearch = Simulation::getCurrent()->getNeighborhoodSearch();
  neighborhoodSearch->set_active(true);
  neighborhoodSearch->find_neighbors();

  // First, need to set the artificial pressure of each rigid body particle to be 0
  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();
  Real dt = TimeManager::getCurrent()->getTimeStepSize();

  // get_delta_time(1,

  #pragma omp parallel for
  forall_rigid_particles(
      if (rbo->isDynamic() || rbo->isAnimated()) {
        vel[R][r] = vel_R + omega_R.cross(r_r);
      }

      bool has_contact = false; 
      // Compute the artificial density of each rigid particle
      forall_rigid_neighbors(
        if(R_k != R)
          has_contact = true;
      );
      if(has_contact)
      {
        density[R][r] = vol0[R][r] * this->W_zero();
        forall_rigid_neighbors(
          density[R][r] += vol0[R_k][k] * this->W(x_r - x_k); // based on SPH rules
        );
        vol[R][r] = density0[R][r] * vol0[R][r] / density[R][r];
      }

      pressure[R][r] = 0.;
      vel_rr[R][r].setZero(););

  clearGradientBeforeSolver();
}

void RigidContactSolver::beforePenaltyInitialize()
{
  Simulation *sim = Simulation::getCurrent();
  density_error_thresh = sim->getRRContactDensityErrorThresh(); // 1 %
  //
  NeighborhoodSearch *neighborhoodSearch = Simulation::getCurrent()->getNeighborhoodSearch();
  neighborhoodSearch->set_active(true);
  neighborhoodSearch->find_neighbors();

  // First, need to set the artificial pressure of each rigid body particle to be 0
  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();
  Real dt = TimeManager::getCurrent()->getTimeStepSize();

  num_contacts = 0; 
  m_particle_indices_in_contact.clear();

  forall_rigid_particles(
      if (rbo->isDynamic() || rbo->isAnimated()) {
        vel[R][r] = vel_R + omega_R.cross(r_r);
      }

      bool has_contact = false; 
      // Compute the artificial density of each rigid particle
      forall_rigid_neighbors(
        if(R_k != R)
          has_contact = true;
      );
      if(has_contact)
      {
        density[R][r] = vol0[R][r] * this->W_zero();
        forall_rigid_neighbors(
          density[R][r] += vol0[R_k][k] * this->W(x_r - x_k); // based on SPH rules
        );
        vol[R][r] = density0[R][r] * vol0[R][r] / density[R][r];
        num_contacts += 1; 
        m_particle_indices_in_contact.push_back({R, r});
      }
  );
}

void RigidContactSolver::solveRigidContact()
{
  Simulation *sim = Simulation::getCurrent();

  if(use_penalty_force)
  {
    beforePenaltyInitialize();
    solveRigidContactPenalty(); 
    for(int R = 0; R<sim->numberOfBoundaryModels(); R++)
    {
      sim->getSimulatorBase()->getBoundarySimulator()->updateBoundaryForces();
    }
    if (sim->useRRContactGradient())
    {
      update_rigid_body_gradient_manager(); 
      //// START_TIMING("perform chain rule after contact");
      //perform_chain_rule_after_contact();
      //// STOP_TIMING_PRINT
    }
  }
  else
  {
    beforeIterationInitialize();
    // Enter the solver iterations
    Real avg_density_err = 1.0;
    unsigned iter_count = 0;

    while (avg_density_err > density_error_thresh && iter_count < sim->getRRContactSolverMaxIter())
    {
      iter_count++;

      // START_TIMING("solveRigidContactIteration");
      avg_density_err = solveRigidContactIteration(density_error_thresh);

      //if (sim->useRRContactGradient())
        //update_rigid_body_gradient_manager(); 
      // STOP_TIMING_PRINT

      if (sim->isDebugRigidContact())
        LOG_INFO << Utilities::RedHead() << "[ " << iter_count << " ] avg_density_err = " << avg_density_err << Utilities::YellowTail();
    }
    if (iter_count > 1)
    {
      if (avg_density_err > density_error_thresh)
        LOG_INFO << Utilities::RedHead() << "[ " << iter_count << " ] avg_density_err = " << avg_density_err << Utilities::YellowTail();

      use_friction = (sim->getFrictionCoeff() > 1e-5);
      if (use_friction)
      {
        // START_TIMING("solveRigidContactIteration");
        solveFriction();
        // STOP_TIMING_PRINT
      }

      // ----------------
      if (sim->useRRContactGradient())
      {
        update_rigid_body_gradient_manager(); 
        //// START_TIMING("perform chain rule after contact");
        //perform_chain_rule_after_contact();
        //// STOP_TIMING_PRINT
      }
      // ----------------
    }

    // TODO: move this into rbGradientManager
    // START_TIMING("perform chain rule");
    //perform_chain_rule();
    // STOP_TIMING_PRINT
  }
}

void RigidContactSolver::solveRigidContactPenalty()
{
  Simulation *sim = Simulation::getCurrent();

  Real contact_stiffness = sim->getBetaScaleFactor(); 

  for(int R = 0; R < sim->numberOfBoundaryModels(); R++)
  {
    for(int RR = 0; RR < sim->numberOfBoundaryModels(); RR++)
    {
        grad_net_force_to_x_before_contact[R][RR].setZero();
        grad_net_force_to_omega_before_contact[R][RR].setZero();
        grad_net_force_to_q_before_contact[R][RR].setZero();
        grad_net_force_to_vel_before_contact[R][RR].setZero();

        grad_net_torque_to_vel_before_contact[R][RR].setZero(); 
        grad_net_torque_to_omega_before_contact[R][RR].setZero(); 
        grad_net_torque_to_x_before_contact[R][RR].setZero(); 
        grad_net_torque_to_q_before_contact[R][RR].setZero(); 
    }
  }

  if(sim->isDebugRigidContact())
    LOG_INFO << Utilities::GreenHead() << "Penalty: num_contacts = " << num_contacts << Utilities::GreenTail();
  // ------------------------------------------------------------------------
  forall_rigid_particles_in_contact(
    Vector3r normal_r = x_r;
    Vector3r vel_rel_r = vel[R][r]; 

    Vector3r sum_x = Vector3r::Zero(); // for computing normal 
    Real sum_w = 0.; 
    Vector3r sum_vel_k = Vector3r::Zero();  // for computing relative velocity
    Vector3r avg_r_k = Vector3r::Zero(); // for computing gradient of friciton

    Vector3r grad_density_to_x = Vector3r::Zero();
    Vector4r grad_density_to_q = Vector4r::Zero();
    Vector4r grad_density_to_q_Rk = Vector4r::Zero();
    
    int RR = R; 
    forall_rigid_neighbors(
      Real w = this->W(x_r - x_k);
      if(R == R_k)
      {
        sum_x += x_k * w;
        sum_w += w; 
      }
      else { // another rigid body 
        RR = R_k; 
        sum_vel_k += vel[R_k][k] * w; 
        avg_r_k += r_k * w;

        grad_density_to_x += vol0[R_k][k] * sim->gradW(x_r - x_k);
        // Here grad_density_to_x_Rk = - grad_density_to_x; 
        grad_density_to_q += vol0[R_k][k] * sim->gradW(x_r - x_k).transpose() * get_grad_Rqp_to_q(rbo->getRotation(), r_r0);
        grad_density_to_q_Rk += vol0[R_k][k] * sim->gradW(x_r - x_k).transpose() * (-1.) * get_grad_Rqp_to_q(rbo_k->getRotation(), r_k0);
      }
    )
    normal_r -= sum_x / sum_w; 
    vel_rel_r -= sum_vel_k / sum_w; 
    avg_r_k /= sum_w;

    auto max = [](const Real a, const Real b){ return a > b ? a : b; };
    const Real thresh = 1.0;  
    if (density[R][r] / density0[R][r] > thresh) 
    {
      Vector3r normal_force = - contact_stiffness * max((density[R][r] / density0[R][r] - thresh), 0) * normal_r;  
      Vector3r unit_vel_rel_r = vel_rel_r.normalized(); 
      Vector3r friction_force = - sim->getFrictionCoeff() * normal_force.norm() * unit_vel_rel_r;

      rbo->addForce(normal_force + friction_force);
      rbo->addTorque(r_r.cross(normal_force + friction_force));

      // ----------------------------------------------------------
      //  Gradient Computation  
      // ----------------------------------------------------------
      Vector3r grad_normal_force_to_density = - contact_stiffness * normal_r;

      Matrix3r grad_normal_force_to_x = grad_normal_force_to_density * grad_density_to_x.transpose(); 
      Matrix34r grad_normal_force_to_q = grad_normal_force_to_density * grad_density_to_q.transpose(); 
      Matrix34r grad_normal_force_to_q_Rk = grad_normal_force_to_density * grad_density_to_q_Rk.transpose(); 

      if(sim->isDebugRigidContact())
      {
        LOG_INFO << "grad_normal_force_to_x = \n" << grad_normal_force_to_x; 
        LOG_INFO << Utilities::CyanHead() <<  "grad_density_to_x = \n" << grad_density_to_x.transpose(); 
        LOG_INFO << Utilities::YellowHead() <<  "grad_normal_force_to_density = \n" 
                << grad_normal_force_to_density.transpose() << Utilities::YellowTail(); 
      }

      // ----------------------------------------------------------
      Vector3r grad_friction_force_to_density = - sim->getFrictionCoeff() * 
          vel_rel_r.normalized() * normal_force.transpose() / normal_force.norm() * grad_normal_force_to_density; 

      Matrix3r grad_friction_force_to_vel_r = - sim->getFrictionCoeff() * normal_force.norm() / vel_rel_r.norm() *
                                              (Matrix3r::Identity() - unit_vel_rel_r * unit_vel_rel_r.transpose()); 
      Matrix3r grad_friction_force_to_vel = grad_friction_force_to_vel_r;  
      // here we assume friction = - mu * normal * (v_r - avg_v_rk)
      // so grad_friction_force_to_vel_rk = - grad_friction_to_vel_r;

      Matrix3r grad_friction_force_to_omega = grad_friction_force_to_vel_r * skewMatrix(r_r).transpose();  
      Matrix3r grad_friction_force_to_omega_Rk = - grad_friction_force_to_vel_r * skewMatrix(avg_r_k).transpose();  
      
      Matrix3r grad_friction_force_to_x = grad_friction_force_to_density * grad_density_to_x.transpose(); 
      Matrix34r grad_friction_force_to_q = grad_friction_force_to_density * grad_density_to_q.transpose(); 
      Matrix34r grad_friction_force_to_q_Rk = grad_friction_force_to_density * grad_density_to_q_Rk.transpose(); 

      // ----------------------------------------------------------
      Vector3r f = normal_force + friction_force; 
      Matrix3r grad_f_to_x = grad_normal_force_to_x + grad_friction_force_to_x;
      grad_net_force_to_x_before_contact[R][R] += grad_f_to_x; 
      grad_net_torque_to_x_before_contact[R][R] += skewMatrix(r_r) * grad_f_to_x;

      Matrix34r grad_f_to_q = grad_normal_force_to_q + grad_friction_force_to_q;
      grad_net_force_to_q_before_contact[R][R] += grad_f_to_q;
      grad_net_torque_to_q_before_contact[R][R] += skewMatrix(r_r) * grad_f_to_q 
                      + skewMatrix(f).transpose() * get_grad_Rqp_to_q(rbo->getRotation(), r_r0);

      grad_net_force_to_vel_before_contact[R][R] += grad_friction_force_to_vel; 
      grad_net_force_to_omega_before_contact[R][R] += grad_friction_force_to_omega; 

      if(RR != R)
      {
        grad_net_force_to_vel_before_contact[R][RR] += - grad_friction_force_to_vel; 
        grad_net_force_to_omega_before_contact[R][RR] += grad_friction_force_to_omega_Rk; 
      
        grad_net_force_to_x_before_contact[R][RR] += - grad_f_to_x;
        grad_net_torque_to_x_before_contact[R][RR] += - skewMatrix(r_r) * grad_f_to_x;

        Matrix34r grad_f_to_q_Rk = grad_normal_force_to_q_Rk + grad_friction_force_to_q_Rk; 
        grad_net_force_to_q_before_contact[R][RR] += grad_f_to_q_Rk; 
        grad_net_torque_to_q_before_contact[R][RR] += skewMatrix(r_r) * grad_f_to_q_Rk;
      }
    }
     
  );

}

Real RigidContactSolver::step()
{
  return solveRigidContactIteration(density_error_thresh);
}

Real RigidContactSolver::solveRigidContactIteration(const Real density_error_thresh)
{
  Simulation *sim = Simulation::getCurrent();
  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();
  Real dt = TimeManager::getCurrent()->getTimeStepSize();
  Real total_density_error = 0.;
  num_contacts = 0;
  m_particle_indices_in_contact.clear();

  // get the fluid-rigid coupling forces and torques,
  // update the velocities of rigidbodies
  // and then clear the stored forces and torques
  sim->getSimulatorBase()->getBoundarySimulator()->updateVelocity();

  // first, traverse all rigid body objects to compute s_r
  Real avg_error = 0.;

  // START_TIMING("compute s_r");
  //#pragma omp parallel
  {
    //#pragma omp for
    forall_rigid_particles(
      div_v_s[R][r] = 0.; // Eq. (9)
      bool has_contact = false; 

      forall_rigid_neighbors(
        if(R_k != R)
        {
          Real mass_k = vol[R_k][k] * density[R_k][k];
          Vector3r x_rk = x_r - x_k; // We cannot use r = x_r - x_k here
          Vector3r delta_vel = vel[R_k][k] - vel[R][r];
          Vector3r w = this->gradW(x_rk);

          // Note: We do not add density[R][r] here since in computing s, we need to compute density[R][r] * div_v^s_r
          div_v_s[R][r] += mass_k * delta_vel.transpose() * w; // Eq.(9) in the paper
        
          has_contact = true;
        }
      )
      s[R][r] = (density0[R][r] - density[R][r]) / dt + div_v_s[R][r];

      if (has_contact){ // TODO: can be accelerated by storing the indices of rigid particles in contact, and only traverse them later 
//PRAGMA_OMP_ATOMIC
        num_contacts += 1;
//PRAGMA_OMP_ATOMIC
        total_density_error += fabs(s[R][r]) / density0[R][r];
        m_particle_indices_in_contact.push_back({R, r}); 
      }
    )
    // STOP_TIMING_PRINT

    // if the source terms of all rigid particles are already lower than the thresh, there is no contact
  }
  avg_error = (total_density_error / n_total_rigid_particles);
  if (num_contacts == 0)
    avg_error = 0.0;
  if (avg_error < density_error_thresh)
    return avg_error; 

  for (int R = 0; R < n_rigid_obj; R++)
  {
    vel_rr_R[R] = Vector3r::Zero();
    omega_rr_R[R] = Vector3r::Zero();
  }

  // --------------------------------------------------------------------------------------------------------------
  //  compute gradient
  // --------------------------------------------------------------------------------------------------------------
  clearGradientDuringIteration();

  forall_rigid_particles_in_contact(
    forall_rigid_neighbors(
      Real mass_k = vol[R_k][k] * density[R_k][k];
      Vector3r x_rk = x_r - x_k; // We cannot use r = x_r - x_k here
      Vector3r delta_vel = vel[R_k][k] - vel[R][r];
      Vector3r grad_w = sim->gradW(x_rk);

      if(rbo->isDynamic())
      {
        // What is the relationship between vel[R][r] and vel_R? : vel[R][r] = vel_R + omega_R.cross(r_r)
        // x[R][r] = x_R + R(q_R) * r_r0
        Vector3r grad_to_x_rk = mass_k * delta_vel.transpose() * get_grad_gradW_to_x_rk(x_rk, grad_w);  

        grad_s_to_vel_before_contact[R][r][R] += mass_k * ( grad_w.transpose() * (- Matrix3r::Identity() ));
        grad_s_to_omega_before_contact[R][r][R] += mass_k * ( grad_w.transpose() * (- skewMatrix(r_r).transpose() ) );
        grad_s_to_x_before_contact[R][r][R] += grad_to_x_rk; 
        grad_s_to_q_before_contact[R][r][R] += grad_to_x_rk.transpose() * get_grad_Rqp_to_q(rbo->getRotation(), r_r0);

        if(rbo_k->isDynamic())
        {
          grad_s_to_vel_before_contact[R][r][R_k] += mass_k * ( grad_w.transpose() * (Matrix3r::Identity() ));
          grad_s_to_omega_before_contact[R][r][R_k] += mass_k * ( grad_w.transpose() * ( skewMatrix(r_k).transpose() ) );
          grad_s_to_x_before_contact[R][r][R_k] += grad_to_x_rk * (-1.);
          grad_s_to_q_before_contact[R][r][R_k] += grad_to_x_rk.transpose() * (-1.) * get_grad_Rqp_to_q(rbo_k->getRotation(), r_k0);
        }
      }
    );
  );
  // end of computing gradient ------------------------------------------------------------------------------------
  // ------------------------------------------------------------------------------------
//#pragma omp parallel
  {
    // ------------------------------------------------------------------------------------
     //To compute the rhs of eq. (8), we need two loops
    // First loop: we compute the pressure gradient as the pressure force of each rigid particle
    // We also accumulate the velocity change of rigid body caused by the pressure force
    // get_delta_time(3,

// START_TIMING("compute nabla_p_r");
//#pragma omp for
      //forall_rigid_particles_in_contact(
      for (int i = 0; i < m_particle_indices_in_contact.size(); i++)\
      {
        auto index = m_particle_indices_in_contact[i];
        unsigned R = index.first;
        unsigned r = index.second;
        
        auto bm = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(R)); 
        auto rbo = bm->getRigidBodyObject(); 
        auto n_particles = bm->numberOfParticles();
        auto vel_R = rbo->getVelocity(); 
        auto omega_R = rbo->getAngularVelocity(); 
        auto mass_R = rbo->getMass();
        auto inertia_R = rbo->getInertiaTensorWorld(); 
        
        auto x_r = bm->getPosition(r);  
        auto r_r = x_r - rbo->getPosition(); 
        auto r_r0 = bm->getPosition0(r); 
        // ---------------------------------------------------------

        Vector3r nabla_p_r = Vector3r::Zero();
        b_from_v_rr_r[R][r] = Vector3r::Zero();
        Matrix3r K_rr = 1. / mass_R * Matrix3r::Identity() - skewMatrix(r_r) * rbo->getInertiaTensorInverseWorld() * skewMatrix(r_r);
        //if (sim->isDebugRigidContact()) // Note: Only for two particles scene
        //{
          //K_rr = 1. / mass_R * Matrix3r::Identity();
        //}

        //forall_rigid_neighbors(

        for (unsigned int pid = sim->numberOfFluidModels(); pid < sim->numberOfPointSets(); pid++) \
        { \
          BoundaryModel_Akinci2012 *bm_neighbor = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModelFromPointSet(pid)); \
          for (unsigned int kk = 0; kk < sim->numberOfNeighbors(sim->numberOfFluidModels()+R, pid, r); kk++) \
          { 
            const unsigned int k = sim->getNeighbor(sim->numberOfFluidModels()+R, pid, r, kk); \
            const unsigned R_k = pid - sim->numberOfFluidModels(); \
            const Vector3r &x_k = bm_neighbor->getPosition(k); \
            auto rbo_k = bm_neighbor->getRigidBodyObject(); \
            const Vector3r r_k = x_k - rbo_k->getPosition(); \
            const Vector3r r_k0 = bm_neighbor->getPosition0(k);\

          if (R_k != R) {
            Real mass_k = vol[R_k][k] * density[R_k][k];
            Vector3r grad_w = this->gradW(x_r - x_k);
            Vector3r norm_x_kr = (x_k - x_r) / (x_k - x_r).norm();
            const Real tmp_pressure = pressure[R][r] / (density[R][r] * density[R][r]) + pressure[R_k][k] / (density[R_k][k] * density[R_k][k]);

            // FIXME: do we need to consider the sample from W(x_r - x_r) here? - No. We do not need to consider that in computing forces.
            nabla_p_r += density[R][r] * mass_k * tmp_pressure * grad_w;

            // for computing b_r
            b_from_v_rr_r[R][r] += K_rr * vol[R][r] * mass_k / density[R][r] * grad_w;

            // --------------------------------------------------------------------------------------------------------------
            //  compute gradient
            // --------------------------------------------------------------------------------------------------------------
            if(rbo->isDynamic())
            {
              auto get_grad_nabla_p_r_to_vel = [&](const unsigned RR)-> Matrix3r
              {
                return density[R][r] * mass_k * (grad_w *
                (grad_pressure_to_vel_before_contact[R][r][RR] / (density[R][r] * density[R][r]) +
                grad_pressure_to_vel_before_contact[R_k][k][RR] / (density[R_k][k] * density[R_k][k]) ).transpose());
              };

              auto get_grad_nabla_p_r_to_omega = [&](const unsigned RR) -> Matrix3r
              {
                return density[R][r] * mass_k * (grad_w *
                (grad_pressure_to_omega_before_contact[R][r][RR] / (density[R][r] * density[R][r]) +
                grad_pressure_to_omega_before_contact[R_k][k][RR] / (density[R_k][k] * density[R_k][k]) ).transpose());
              };

              // --------------------------------------------------------------------------------------------------------------
            
              Matrix3r grad_to_x_rk = tmp_pressure * get_grad_gradW_to_x_rk(x_r - x_k, grad_w);

              auto get_grad_nabla_p_r_to_x = [&](const unsigned RR) -> Matrix3r
              {
                Matrix3r grad1 = grad_w * (grad_pressure_to_x_before_contact[R][r][RR] / (density[R][r] * density[R][r]) +
                          grad_pressure_to_x_before_contact[R_k][k][RR] / (density[R_k][k] * density[R_k][k]) ).transpose();
                if(RR == R)
                  return density[R][r] * mass_k * (grad1 + grad_to_x_rk);
                else if(RR == R_k) 
                  return density[R][r] * mass_k * (grad1 - grad_to_x_rk);
                else 
                  return Matrix3r::Zero(); 
              };
              auto get_grad_nabla_p_r_to_q = [&](const unsigned RR) -> Matrix34r
              {
                Matrix34r grad1 = grad_w * (grad_pressure_to_q_before_contact[R][r][RR] / (density[R][r] * density[R][r]) +
                          grad_pressure_to_q_before_contact[R_k][k][RR] / (density[R_k][k] * density[R_k][k]) ).transpose();
                if(RR == R)
                  return density[R][r] * mass_k * (grad1 + grad_to_x_rk * get_grad_Rqp_to_q(rbo->getRotation(), r_r0) );
                else if(RR == R_k) 
                  return density[R][r] * mass_k * (grad1 - grad_to_x_rk * get_grad_Rqp_to_q(rbo_k->getRotation(), r_k0));
                else 
                  return Matrix34r::Zero(); 
              };
              // --------------------------------------------------------------------------------------------------------------

              grad_nabla_p_r_to_vel_before_contact[R][r][R] += get_grad_nabla_p_r_to_vel(R);
              grad_nabla_p_r_to_omega_before_contact[R][r][R] += get_grad_nabla_p_r_to_omega(R);
              grad_nabla_p_r_to_x_before_contact[R][r][R] += get_grad_nabla_p_r_to_x(R);
              grad_nabla_p_r_to_q_before_contact[R][r][R] += get_grad_nabla_p_r_to_q(R);

              // This gradient represents how the perturbation of velocity of rigid body R_k affects the nabla_p_r of rigid particle r:
              if(rbo_k->isDynamic())
              {
                grad_nabla_p_r_to_vel_before_contact[R][r][R_k] += get_grad_nabla_p_r_to_vel(R_k);
                grad_nabla_p_r_to_omega_before_contact[R][r][R_k] += get_grad_nabla_p_r_to_omega(R_k);
                grad_nabla_p_r_to_x_before_contact[R][r][R_k] += get_grad_nabla_p_r_to_x(R_k);
                grad_nabla_p_r_to_q_before_contact[R][r][R_k] += get_grad_nabla_p_r_to_q(R_k);
              }
            }
            // end of computing gradient --------------------------------------------------------------------------------------------------------------
          }
        //)
        }}

        nabla_pressure[R][r] = nabla_p_r;
        if (rbo->isDynamic())
        {
          Vector3r delta_vel_rr_R = -dt * 1.0 / mass_R * vol[R][r] * nabla_p_r;
          Vector3r delta_omega_rr_R = -dt * rbo->getInertiaTensorInverseWorld() * vol[R][r] * r_r.cross(nabla_p_r);
          //if (sim->isDebugRigidContact()) // Note: Only for two particles scene
          //{
            //delta_omega_rr_R = Vector3r::Zero();
          //}
          
//PRAGMA_OMP_ATOMIC
          vel_rr_R[R][0] += delta_vel_rr_R[0];
//PRAGMA_OMP_ATOMIC
          vel_rr_R[R][1] += delta_vel_rr_R[1];
//PRAGMA_OMP_ATOMIC
          vel_rr_R[R][2] += delta_vel_rr_R[2];
//PRAGMA_OMP_ATOMIC
          omega_rr_R[R][0] += delta_omega_rr_R[0];
//PRAGMA_OMP_ATOMIC
          omega_rr_R[R][1] += delta_omega_rr_R[1];
//PRAGMA_OMP_ATOMIC
          omega_rr_R[R][2] += delta_omega_rr_R[2];

          // --------------------------------------------------------------------------------------
          // compute gradient 
          // --------------------------------------------------------------------------------------
          for(int RR = 0; RR < n_rigid_obj; RR++)
          {
            auto rbo_RR = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(RR))->getRigidBodyObject();
            if(rbo_RR->isDynamic())
            {
              grad_net_force_to_vel_before_contact[R][RR] += -vol[R][r] * grad_nabla_p_r_to_vel_before_contact[R][r][RR];
              grad_net_force_to_omega_before_contact[R][RR] += -vol[R][r] * grad_nabla_p_r_to_omega_before_contact[R][r][RR];
              grad_net_force_to_x_before_contact[R][RR] += -vol[R][r] * grad_nabla_p_r_to_x_before_contact[R][r][RR];
              grad_net_force_to_q_before_contact[R][RR] += -vol[R][r] * grad_nabla_p_r_to_q_before_contact[R][r][RR];

              grad_net_torque_to_vel_before_contact[R][RR] += -vol[R][r] * skewMatrix(r_r) * grad_nabla_p_r_to_vel_before_contact[R][r][RR];
              grad_net_torque_to_omega_before_contact[R][RR] += -vol[R][r] * skewMatrix(r_r) * grad_nabla_p_r_to_omega_before_contact[R][r][RR];
              grad_net_torque_to_x_before_contact[R][RR] += -vol[R][r] * skewMatrix(r_r) * grad_nabla_p_r_to_x_before_contact[R][r][RR];
              grad_net_torque_to_q_before_contact[R][RR] += -vol[R][r] * skewMatrix(r_r) * grad_nabla_p_r_to_q_before_contact[R][r][RR];

              // --------------------------------------------------------------------------------------
          //
              grad_vel_rr_R_to_vel_before_contact[R][RR] += - dt * 1.0 / mass_R * vol[R][r] * grad_nabla_p_r_to_vel_before_contact[R][r][RR];
              grad_vel_rr_R_to_omega_before_contact[R][RR] += - dt * 1.0 / mass_R * vol[R][r] * grad_nabla_p_r_to_omega_before_contact[R][r][RR];
              grad_vel_rr_R_to_x_before_contact[R][RR] += - dt * 1.0 / mass_R * vol[R][r] * grad_nabla_p_r_to_x_before_contact[R][r][RR];
              grad_vel_rr_R_to_q_before_contact[R][RR] += - dt * 1.0 / mass_R * vol[R][r] * grad_nabla_p_r_to_q_before_contact[R][r][RR];

              // TODO: need to consider the gradient of inertia tensor
              grad_omega_rr_R_to_vel_before_contact[R][RR] += -dt * rbo->getInertiaTensorInverseWorld() *
                vol[R][r] * skewMatrix(r_r) * grad_nabla_p_r_to_vel_before_contact[R][r][RR];
              grad_omega_rr_R_to_omega_before_contact[R][RR] += -dt * rbo->getInertiaTensorInverseWorld() *
                vol[R][r] * skewMatrix(r_r) * grad_nabla_p_r_to_omega_before_contact[R][r][RR];
              grad_omega_rr_R_to_x_before_contact[R][RR] += -dt * rbo->getInertiaTensorInverseWorld() *
                vol[R][r] * skewMatrix(r_r) * grad_nabla_p_r_to_x_before_contact[R][r][RR];
              grad_omega_rr_R_to_q_before_contact[R][RR] += -dt * rbo->getInertiaTensorInverseWorld() *
                vol[R][r] * skewMatrix(r_r) * grad_nabla_p_r_to_q_before_contact[R][r][RR];

              // --------------------------------------------------------------------------------------
              if(sim->isDebugRigidContact())
              {
                LOG_INFO << Utilities::RedHead() << "grad_net_force_to_vel_before_contact[" << R << "][" << RR << "] = " << grad_net_force_to_vel_before_contact[R][RR] << Utilities::RedTail();
                LOG_INFO << Utilities::YellowHead() << "grad_vel_rr_R_to_vel_before_contact[" << R << "][" << RR << "] = " << grad_vel_rr_R_to_vel_before_contact[R][RR] << Utilities::RedTail();
              }
            }
          }
          // end of computing gradient --------------------------------------------------------------------------------------
        }
    //)
    }
    // STOP_TIMING_PRINT

    // -------------------------------------------------------------------------

    // START_TIMING("update rbo vel");
    //  update the velocity and angular velocity of rigid body R
    for (int R = 0; R < n_rigid_obj; R++)
    {
      auto bm = static_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModel(R));
      auto rbo = bm->getRigidBodyObject();

      if (rbo->isDynamic())
      {
        rbo->setVelocity(rbo->getVelocity() + vel_rr_R[R]);
        rbo->setAngularVelocity(rbo->getAngularVelocity() + omega_rr_R[R]);
      }
    }
// STOP_TIMING_PRINT

// the second loop
// We compute the velocity change of each rigid particle due to rigid-rigid contact
// get_delta_time(4,

// START_TIMING("second loop");
// START_TIMING("compute vel[R][r]");
//#pragma omp for
    forall_rigid_particles_in_contact(
        if (rbo->isDynamic()) {
          // if(is_in_contact(R, r))
          //{
          auto x_r = bm->getPosition(r);
          auto r_r = x_r - rbo->getPosition();
          vel_rr[R][r] = vel_rr_R[R] + omega_rr_R[R].cross(r_r); // compute the predicted rigid velocity, alg line 8
          vel[R][r] += vel_rr[R][r]; // directly update the velocity of each rigid particle here. ;
    
          // --------------------------------------------------------------------------------------
          // compute gradient 
          // --------------------------------------------------------------------------------------
          for(int RR = 0; RR < n_rigid_obj; RR++)
          {
            auto rbo_RR = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(RR))->getRigidBodyObject();
             if(rbo_RR->isDynamic())
            {
              // vel_rr = vel_rr_R + omeag_rr_R.cross(r_r)
              // TODO: here we need to consider the gradient of r_r: r_r = R(q) * r_r0
              grad_vel_rr_to_vel_before_contact[R][r][RR] = grad_vel_rr_R_to_vel_before_contact[R][RR] + skewMatrix(r_r).transpose() * grad_omega_rr_R_to_vel_before_contact[R][RR];
              grad_vel_rr_to_omega_before_contact[R][r][RR] = grad_vel_rr_R_to_omega_before_contact[R][RR] + skewMatrix(r_r).transpose() * grad_omega_rr_R_to_omega_before_contact[R][RR];
              grad_vel_rr_to_x_before_contact[R][r][RR] = grad_vel_rr_R_to_x_before_contact[R][RR] + skewMatrix(r_r).transpose() * grad_omega_rr_R_to_x_before_contact[R][RR];
              grad_vel_rr_to_q_before_contact[R][r][RR] = grad_vel_rr_R_to_q_before_contact[R][RR] + skewMatrix(r_r).transpose() * grad_omega_rr_R_to_q_before_contact[R][RR];
            }
          }
          // end of computing gradient --------------------------------------------------------------------------------------
        })
// STOP_TIMING_PRINT

// ----------------------------------------------------------------------

// START_TIMING("compute div_vel_rr");
//#pragma omp for
    forall_rigid_particles_in_contact(
        div_vel_rr[R][r] = 0.;
        b[R][r] = 0.;
          forall_rigid_neighbors(
              if(R_k != R)
              {
                Real mass_k = vol[R_k][k] * density[R_k][k];
                Vector3r grad_w = this->gradW(x_r - x_k);
                Vector3r delta_vel_rr = vel_rr[R_k][k] - vel_rr[R][r];
                div_vel_rr[R][r] += mass_k * delta_vel_rr.transpose() * grad_w;
                // ---------------------------- compute b_r --------------------------------
                Vector3r b_from_v_rr_k = Vector3r::Zero();

                // Traverse again the neighbor of particle r
                for (unsigned int pid = sim->numberOfFluidModels(); pid < sim->numberOfPointSets(); pid++) {
                  BoundaryModel_Akinci2012 *bm_neighbor = static_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModelFromPointSet(pid));
                  for (unsigned int kk = 0; kk < sim->numberOfNeighbors(sim->numberOfFluidModels() + R, pid, r); kk++)
                  {
                    const unsigned int j = sim->getNeighbor(sim->numberOfFluidModels() + R, pid, r, kk);
                    const unsigned R_j = pid - sim->numberOfFluidModels();
                    if (R_j == R_k)
                    {
                      Vector3r x_j = bm_neighbor->getPosition(j);
                      auto rbo_j = bm_neighbor->getRigidBodyObject();
                      Vector3r r_j = x_j - rbo_j->getPosition();

                      Matrix3r K_kj = 1.0 / rbo_j->getMass() * Matrix3r::Identity() - skewMatrix(r_k) * rbo_j->getInertiaTensorInverseWorld() * skewMatrix(r_j);
                      b_from_v_rr_k += vol[R_j][j] * K_kj * density[R_j][j] * vol[R][r] / density[R][r] * this->gradW(x_j - x_r);
                    }
                  }
                }

                // for relaxed jacobi solver to use
                b[R][r] += -vol[R_k][k] * density[R_k][k] * dt *
                           (b_from_v_rr_r[R][r] - b_from_v_rr_k).transpose() * this->gradW(x_r - x_k);
                //b[R][r] += -vol[R_k][k] * density[R_k][k] * dt *
                           //(b_from_v_rr_r[R][r] ).transpose() * this->gradW(x_r - x_k);

              // --------------------------------------------------------------------------------------
              // compute gradient 
              // --------------------------------------------------------------------------------------
              if(rbo->isDynamic())
              {
                // --------------------------------------------------------------------------------------------------------------------------
                auto get_grad_div_vel_rr_to_vel = [&](const unsigned RR) -> Vector3r
                {
                  return mass_k * grad_w.transpose() * (grad_vel_rr_to_vel_before_contact[R_k][k][RR] - grad_vel_rr_to_vel_before_contact[R][r][RR]);
                };
                auto get_grad_div_vel_rr_to_omega = [&](const unsigned RR)-> Vector3r
                {
                  return mass_k * grad_w.transpose() * (grad_vel_rr_to_omega_before_contact[R_k][k][RR] - grad_vel_rr_to_omega_before_contact[R][r][RR]);
                };

                Vector3r grad_to_x_rk = delta_vel_rr.transpose() * get_grad_gradW_to_x_rk(x_r - x_k, grad_w);

                auto get_grad_div_vel_rr_to_x = [&](const unsigned RR)-> Vector3r
                {
                  Vector3r grad1 = (grad_w.transpose() * (grad_vel_rr_to_x_before_contact[R_k][k][RR] - grad_vel_rr_to_x_before_contact[R][r][RR]));
                  if(RR == R)
                    return mass_k * (grad1 + grad_to_x_rk);
                  else if(RR == R_k)
                    return mass_k * (grad1 - grad_to_x_rk);
                  else 
                    return Vector3r::Zero();
                };
                auto get_grad_div_vel_rr_to_q = [&](const unsigned RR)-> Vector4r
                {
                  Vector4r grad1 = (grad_w.transpose() * (grad_vel_rr_to_q_before_contact[R_k][k][RR] - grad_vel_rr_to_q_before_contact[R][r][RR])).transpose();
                  if(RR == R)
                    return mass_k * (grad1 + (grad_to_x_rk.transpose() * get_grad_Rqp_to_q(rbo->getRotation(), r_r0)).transpose());
                  else if(RR == R_k)
                    return mass_k * (grad1 - (grad_to_x_rk.transpose() * get_grad_Rqp_to_q(rbo_k->getRotation(), r_k0)).transpose());
                  else 
                    return Vector4r::Zero();
                };
                // --------------------------------------------------------------------------------------------------------------------------

                grad_div_vel_rr_to_vel_before_contact[R][r][R] += get_grad_div_vel_rr_to_vel(R);
                grad_div_vel_rr_to_omega_before_contact[R][r][R] += get_grad_div_vel_rr_to_omega(R);
                grad_div_vel_rr_to_x_before_contact[R][r][R] += get_grad_div_vel_rr_to_x(R);
                grad_div_vel_rr_to_q_before_contact[R][r][R] += get_grad_div_vel_rr_to_q(R);

               if(rbo_k->isDynamic())
                {
                grad_div_vel_rr_to_vel_before_contact[R][r][R_k] += get_grad_div_vel_rr_to_vel(R_k);
                grad_div_vel_rr_to_omega_before_contact[R][r][R_k] += get_grad_div_vel_rr_to_omega(R_k);
                grad_div_vel_rr_to_x_before_contact[R][r][R_k] += get_grad_div_vel_rr_to_x(R_k);
                grad_div_vel_rr_to_q_before_contact[R][r][R_k] += get_grad_div_vel_rr_to_q(R_k);
              }
            }
          }
            // end of computing gradient --------------------------------------------------------------------------------------
          )
        )
// STOP_TIMING_PRINT;

// ----------------------------------------------------------------------
// START_TIMING("update pressure");
    avg_error = 0; 
    Real beta = 0.5 / num_contacts * sim->getBetaScaleFactor();
    //LOG_INFO << Utilities::RedHead() << " beta = " << beta << " , num_contacts = " << num_contacts << Utilities::YellowTail();

//#pragma omp for
    forall_rigid_particles_in_contact(
      //PRAGMA_OMP_ATOMIC
      avg_error += fabs(s[R][r] + div_vel_rr[R][r]) / density0[R][r];

      b[R][r] = (b[R][r] < 0.) ? b[R][r] : 0.;

      // Real beta = 0.5 / num_contacts;

      // Real h = sim->getParticleRadius();
      // if(sim->is2DSimuliation())
      // beta *= vol[R][r] / h / h;
      // else
      // beta *= vol[R][r] / h / h / h;
      if (fabs(b[R][r]) > 1e-16)
      {
        pressure[R][r] += beta / b[R][r] * (s[R][r] + div_vel_rr[R][r]);
        //LOG_INFO << Utilities::RedHead() << " b[" << R << "][" << r << "] = " << b[R][r] << Utilities::YellowTail();
        pressure[R][r] = (pressure[R][r] > 0.) ? pressure[R][r] : 0.; // clamp
      }

      // --------------------------------------------------------------------------------------
      // compute gradient 
      // --------------------------------------------------------------------------------------
      for(int RR = 0; RR < n_rigid_obj; RR++)
      {
        auto rbo_RR = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(RR))->getRigidBodyObject();
        if(rbo_RR->isDynamic())
        {
          if(pressure[R][r] > 0.)
          {
            grad_pressure_to_vel_before_contact[R][r][RR] += beta / b[R][r] * (grad_s_to_vel_before_contact[R][r][RR] + grad_div_vel_rr_to_vel_before_contact[R][r][RR]);
            grad_pressure_to_omega_before_contact[R][r][RR] += beta / b[R][r] * (grad_s_to_omega_before_contact[R][r][RR] + grad_div_vel_rr_to_omega_before_contact[R][r][RR]);
            grad_pressure_to_x_before_contact[R][r][RR] += beta / b[R][r] * (grad_s_to_x_before_contact[R][r][RR] + grad_div_vel_rr_to_x_before_contact[R][r][RR]);
            grad_pressure_to_q_before_contact[R][r][RR] += beta / b[R][r] * (grad_s_to_q_before_contact[R][r][RR] + grad_div_vel_rr_to_q_before_contact[R][r][RR]);
          }
        else
          {
            grad_pressure_to_vel_before_contact[R][r][RR].setZero();
            grad_pressure_to_omega_before_contact[R][r][RR].setZero();
            grad_pressure_to_x_before_contact[R][r][RR].setZero();
            grad_pressure_to_q_before_contact[R][r][RR].setZero();
        }
      }
    }
    // end of computing gradient --------------------------------------------------------------------------------------
  )

}
avg_error /= n_total_rigid_particles;
                // STOP_TIMING_PRINT
  
  // STOP_TIMING_PRINT
  // end of one iteration
  return avg_error;
}

void RigidContactSolver::reset()
{
  Simulation *sim = Simulation::getCurrent();
  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();
  forall_rigid_particles(
      pressure[R][r] = 0.;
      vel[R][r].setZero(););
}

bool RigidContactSolver::is_in_contact(const unsigned R, const unsigned r) const
{
  // Real eps = 0.001;
  // if( (density[R][r] - density0[R][r]) > eps * density0[R][r] ) // if in contact
  bool has_contact = false; 
  Simulation *sim = Simulation::getCurrent();
  auto bm = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(R)); 
  auto rbo = bm->getRigidBodyObject(); 
  auto x_r = bm->getPosition(r);  
  auto r_r = x_r - rbo->getPosition(); 
  auto r_r0 = bm->getPosition0(r); 

  forall_rigid_neighbors(
    if(R_k != R)
      has_contact = true;
  );
  //Real err = fabs(s[R][r]) / density0[R][r]; // FIXME: error here !
  //if (err > density_error_thresh)
  return has_contact;
}

void RigidContactSolver::solveFriction()
{
  if (num_contacts <= 1) // no contact, return
    return;

  Simulation *sim = Simulation::getCurrent();
  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();
  Real dt = TimeManager::getCurrent()->getTimeStepSize();

  for (int R = 0; R < n_rigid_obj; R++)
  {
    auto bm = static_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModel(R));
    auto rbo = bm->getRigidBodyObject();
    auto n_particles = bm->numberOfParticles();
    Real mass_R = rbo->getMass();

    if (rbo->isDynamic())
    {
      for (int r = 0; r < n_particles; r++)
      {
        if (is_in_contact(R, r)) // if in contact
        {
          Vector3r x_r = bm->getPosition(r);
          Vector3r r_r = x_r - rbo->getPosition();
          Vector3r r_r0 = bm->getPosition0(r);

          Vector3r normal_r = x_r;
          Vector3r sum_x_k = Vector3r::Zero();
          Real sum_w = 0.;

          Vector3r vel_rel_r = vel[R][r];
          Vector3r sum_vel_k = Vector3r::Zero();

          Vector3r avg_r_k = Vector3r::Zero();

          // Here we assume R only in contact with single rbo R_k in computing friction, which is reasonable
          int neighbor_rbo_index = -1;

          forall_rigid_neighbors(
              if (R_k != R) {
                Real w = this->W(x_r - x_k);
                sum_x_k += x_k * w;
                sum_w += w;
                sum_vel_k += vel[R_k][k] * w;

                avg_r_k += r_k * w;

                if (neighbor_rbo_index < 0)
                  neighbor_rbo_index = R_k; // Here we assume R only in contact with single rbo R_k in computing friction, which is reasonable
              }

          );

          normal_r -= sum_x_k / sum_w;
          vel_rel_r -= sum_vel_k / sum_w;
          avg_r_k /= sum_w;
          // Note: grad_vel_rel_r_to_vel[R] = Identity
          // grad_vel_rel_r_to_vel[R_k] = -Identity
          // grad_vel_r_to_omega[R] = skewMatrix(r_r).transpose()
          // grad_vel_r_to_omega[R_k] = - \sum_k skewMatrix(r_k).transpose() / neighbor_count = - skewMatrix(avg_r_k).transpose();

          normal_r = -normal_r.normalized(); // normal at particle r points outwards from particle r
          if (vel_rel_r.dot(normal_r) > 0)   // Columb model
          {
            Vector3r vel_rel_normal = vel_rel_r.dot(normal_r) * normal_r;
            Vector3r vel_rel_t = vel_rel_r - vel_rel_normal;
            Vector3r t_r = vel_rel_t.normalized();

            Vector3r friction = sim->getFrictionCoeff() * vel_rel_normal.norm() * -t_r;

            // ----------------------------------------------------------------
            // Matrix3r K_rr = 1. / mass_R * Matrix3r::Identity() - skewMatrix(r_r) * rbo->getInertiaTensorInverseWorld() * skewMatrix(r_r);
            // Vector3r max_bound = - 1.0 / dt / num_contacts / (t_r.transpose() * K_rr* t_r) * vel_rel_t;

            // auto clamp= [](const Vector3r& v, const Vector3r& upper_bounds) {
            // return Vector3r(v.array().min(upper_bounds.array()) );
            // };
            // friction = clamp(friction, max_bound);

            // ----------------------------------------------------------------
            Vector3r delta_vel_R = dt * 1.0 / mass_R * friction;
            Vector3r delta_omega_R = dt * rbo->getInertiaTensorInverseWorld() * r_r.cross(friction);
            rbo->setVelocity(rbo->getVelocity() + delta_vel_R);
            // rbo->setAngularVelocity(rbo->getAngularVelocity() + delta_omega_R);
            //  ----------------------------------------------------------------

            Matrix3r grad_friction_to_vel_rel_normal = sim->getFrictionCoeff() * -t_r * vel_rel_normal.transpose() / vel_rel_normal.norm();
            // grad_vel_rel_normal_to_vel_r = n * n^T * Identity
            Matrix3r grad_vel_rel_normal_to_vel_r = normal_r * normal_r.transpose();
            Matrix3r grad_friction_to_vel_r = grad_friction_to_vel_rel_normal * grad_vel_rel_normal_to_vel_r;

            vector<int> R_list = {R, neighbor_rbo_index};

            for (int RR : R_list)
            {
              if (RR == R)
              {
                // vel_r = vel_R + omega_R.cross(r_r) = vel_R + skew(r_r)^T * omega_R
                grad_friction_to_vel_before_contact[R][RR] = grad_friction_to_vel_r;
                grad_friction_to_omega_before_contact[R][RR] = grad_friction_to_vel_r * skewMatrix(r_r).transpose();
              }
              else if (neighbor_rbo_index == RR && (RR > 0))
              {
                // grad_vel_rel_to_vel_k = - Identity (On average)
                // grad_vel_rel_normal_to_vel_k = n * n^T * - Identity (On average)
                // Matrix3r grad_vel_rel_normal_to_vel_k = - grad_vel_rel_normal_to_vel_k;
                // Matrix3r grad_friction_to_vel_k = grad_friction_to_vel_rel_normal * grad_vel_rel_normal_to_vel_k;

                // Note: in real Columb model, friction need to be clamped, but now we ignore this clamping
                Matrix3r grad_friction_to_vel_k = -grad_friction_to_vel_r;
                grad_friction_to_vel_before_contact[R][RR] = grad_friction_to_vel_k;
                grad_friction_to_omega_before_contact[R][RR] = grad_friction_to_vel_k * skewMatrix(avg_r_k).transpose();
              }

              grad_vel_rr_R_to_vel_before_contact[R][RR] += dt * 1.0 / mass_R * grad_friction_to_vel_before_contact[R][RR];
              grad_vel_rr_R_to_omega_before_contact[R][RR] += dt * 1.0 / mass_R * grad_friction_to_vel_before_contact[R][RR];
              grad_omega_rr_R_to_vel_before_contact[R][RR] += dt * rbo->getInertiaTensorInverseWorld() * skewMatrix(r_r) * grad_friction_to_vel_before_contact[R][RR];
              grad_omega_rr_R_to_omega_before_contact[R][RR] += dt * rbo->getInertiaTensorInverseWorld() * skewMatrix(r_r) * grad_friction_to_omega_before_contact[R][RR];
            }
          }
        }
      }
    }
  }
}

void RigidContactSolver::clearGradientDuringIteration()
{
  Simulation *sim = Simulation::getCurrent();
  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();

  forall_rigid_particles_in_contact(
      for (int RR = 0; RR < n_rigid_obj; RR++)
      {
        // grad_pressure_to_vel[R][r][RR].setZero();
        // grad_pressure_to_omega[R][r][RR].setZero();
        grad_s_to_vel_before_contact[R][r][RR].setZero();
        grad_s_to_omega_before_contact[R][r][RR].setZero();
        grad_s_to_x_before_contact[R][r][RR].setZero();
        grad_s_to_q_before_contact[R][r][RR].setZero();

        grad_div_vel_rr_to_vel_before_contact[R][r][RR].setZero();
        grad_div_vel_rr_to_omega_before_contact[R][r][RR].setZero();
        grad_div_vel_rr_to_x_before_contact[R][r][RR].setZero();
        grad_div_vel_rr_to_q_before_contact[R][r][RR].setZero();

        grad_nabla_p_r_to_vel_before_contact[R][r][RR].setZero();
        grad_nabla_p_r_to_omega_before_contact[R][r][RR].setZero();
        grad_nabla_p_r_to_x_before_contact[R][r][RR].setZero();
        grad_nabla_p_r_to_q_before_contact[R][r][RR].setZero();

        grad_vel_rr_to_vel_before_contact[R][r][RR].setZero();
        grad_vel_rr_to_omega_before_contact[R][r][RR].setZero();
        grad_vel_rr_to_x_before_contact[R][r][RR].setZero();
        grad_vel_rr_to_q_before_contact[R][r][RR].setZero();
      }
  )
    
  for(int R = 0; R < n_rigid_obj; R++)
  {
    for (int RR = 0; RR < n_rigid_obj; RR++)
    {
      grad_vel_rr_R_to_vel_before_contact[R][RR].setZero();
      grad_vel_rr_R_to_omega_before_contact[R][RR].setZero();
      grad_vel_rr_R_to_x_before_contact[R][RR].setZero();
      grad_vel_rr_R_to_q_before_contact[R][RR].setZero();

      grad_omega_rr_R_to_vel_before_contact[R][RR].setZero();
      grad_omega_rr_R_to_omega_before_contact[R][RR].setZero();
      grad_omega_rr_R_to_x_before_contact[R][RR].setZero();
      grad_omega_rr_R_to_q_before_contact[R][RR].setZero();
    }
  }
}

void RigidContactSolver::clearGradientBeforeSolver()
{
  clearGradientDuringIteration();

  Simulation *sim = Simulation::getCurrent();
  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();

  auto clearVec = [](vector<Vector3r> &v)
  {
    std::fill(v.begin(), v.end(), Vector3r::Zero());
  };
  auto clearMat = [](vector<Matrix3r> &v)
  {
    std::fill(v.begin(), v.end(), Matrix3r::Zero());
  };

  for (int R = 0; R < n_rigid_obj; R++)
  {
    auto bm = static_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModel(R));
    unsigned n_particles = bm->numberOfParticles();

    PRAGMA_OMP_FOR
    for (int r = 0; r < n_particles; r++)
    {
      for (int RR = 0; RR < n_rigid_obj; RR++)
      {
        grad_pressure_to_vel_before_contact[R][r][RR].setZero();
        grad_pressure_to_omega_before_contact[R][r][RR].setZero();
        grad_pressure_to_x_before_contact[R][r][RR].setZero();
        grad_pressure_to_q_before_contact[R][r][RR].setZero();
      }
    }

    for (int RR = 0; RR < n_rigid_obj; RR++)
    {
      // grad_vel_rr_R_to_vel[R][RR].setZero();
      // grad_vel_rr_R_to_omega[R][RR].setZero();
      // grad_omega_rr_R_to_vel[R][RR].setZero();
      // grad_omega_rr_R_to_omega[R][RR].setZero();
      grad_friction_to_vel_before_contact[R][RR].setZero();
      grad_friction_to_omega_before_contact[R][RR].setZero();
      grad_friction_to_x_before_contact[R][RR].setZero();
      grad_friction_to_q_before_contact[R][RR].setZero();

      grad_net_force_to_vel_before_contact[R][RR].setZero(); 
      grad_net_force_to_omega_before_contact[R][RR].setZero(); 
      grad_net_force_to_x_before_contact[R][RR].setZero(); 
      grad_net_force_to_q_before_contact[R][RR].setZero(); 
      grad_net_torque_to_vel_before_contact[R][RR].setZero(); 
      grad_net_torque_to_omega_before_contact[R][RR].setZero(); 
      grad_net_torque_to_x_before_contact[R][RR].setZero(); 
      grad_net_torque_to_q_before_contact[R][RR].setZero(); 
    }
  }
}

void RigidContactSolver::update_rigid_body_gradient_manager()
{
  Simulation *sim = Simulation::getCurrent();
  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();
  auto rb_grad_manager = sim->getRigidBodyGradientManager(); 

  for(int R = 0; R < n_rigid_obj; R++)
  {
    if(sim->getBoundaryModel(R)->getRigidBodyObject()->isDynamic() == false)
      continue; 

    for(int RR = 0; RR < n_rigid_obj; RR++)
    {
      if(sim->getBoundaryModel(RR)->getRigidBodyObject()->isDynamic() == false)
        continue; 

      rb_grad_manager->get_grad_net_force_to_vn(R, RR) = grad_net_force_to_vel_before_contact[R][RR]; 
      rb_grad_manager->get_grad_net_force_to_omega_n(R, RR) = grad_net_force_to_omega_before_contact[R][RR]; 
      rb_grad_manager->get_grad_net_force_to_xn(R, RR) = grad_net_force_to_x_before_contact[R][RR]; 
      rb_grad_manager->get_grad_net_force_to_qn(R, RR) = grad_net_force_to_q_before_contact[R][RR]; 

      rb_grad_manager->get_grad_net_torque_to_vn(R, RR) = grad_net_torque_to_vel_before_contact[R][RR]; 
      rb_grad_manager->get_grad_net_torque_to_omega_n(R, RR) = grad_net_torque_to_omega_before_contact[R][RR]; 
      rb_grad_manager->get_grad_net_torque_to_xn(R, RR) = grad_net_torque_to_x_before_contact[R][RR]; 
      rb_grad_manager->get_grad_net_torque_to_qn(R, RR) = grad_net_torque_to_q_before_contact[R][RR]; 
    }
  }
  
}


void RigidContactSolver::performNeighborhoodSearchSort()
{
  if (Simulation::getCurrent()->zSortEnabled())
  {
    Simulation *sim = Simulation::getCurrent();
    const unsigned n_rigid_obj = sim->numberOfBoundaryModels();
    NeighborhoodSearch *neighborhoodSearch = Simulation::getCurrent()->getNeighborhoodSearch();
    for (int R = 0; R < n_rigid_obj; R++)
    {
      auto bm = static_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModel(R));
      auto rbo = bm->getRigidBodyObject();
      if (!rbo->isDynamic() && !rbo->isAnimated() && bm->isSorted())
        continue;
      else
      {
        auto const &d = neighborhoodSearch->point_set(bm->getPointSetIndex());

        d.sort_field(&pressure[R][0]);
        d.sort_field(&nabla_pressure[R][0]);
        d.sort_field(&density[R][0]);
        d.sort_field(&density0[R][0]);
        d.sort_field(&vel[R][0]);
        d.sort_field(&vel_rr[R][0]);
        d.sort_field(&vol[R][0]);
        d.sort_field(&vol0[R][0]);
        d.sort_field(&s[R][0]);
        d.sort_field(&b_from_v_rr_r[R][0]);
        d.sort_field(&b[R][0]);
        d.sort_field(&div_vel_rr[R][0]);
        d.sort_field(&div_v_s[R][0]);

        d.sort_field(&grad_pressure_to_vel_before_contact[R][0]);
        d.sort_field(&grad_pressure_to_omega_before_contact[R][0]);
        d.sort_field(&grad_pressure_to_x_before_contact[R][0]);
        d.sort_field(&grad_pressure_to_q_before_contact[R][0]);

        d.sort_field(&grad_s_to_vel_before_contact[R][0]);
        d.sort_field(&grad_s_to_omega_before_contact[R][0]);
        d.sort_field(&grad_s_to_x_before_contact[R][0]);
        d.sort_field(&grad_s_to_q_before_contact[R][0]);

        d.sort_field(&grad_div_vel_rr_to_vel_before_contact[R][0]);
        d.sort_field(&grad_div_vel_rr_to_omega_before_contact[R][0]);
        d.sort_field(&grad_div_vel_rr_to_x_before_contact[R][0]);
        d.sort_field(&grad_div_vel_rr_to_q_before_contact[R][0]);

        d.sort_field(&grad_nabla_p_r_to_vel_before_contact[R][0]);
        d.sort_field(&grad_nabla_p_r_to_omega_before_contact[R][0]);
        d.sort_field(&grad_nabla_p_r_to_x_before_contact[R][0]);
        d.sort_field(&grad_nabla_p_r_to_q_before_contact[R][0]);

        d.sort_field(&grad_vel_rr_to_vel_before_contact[R][0]);
        d.sort_field(&grad_vel_rr_to_omega_before_contact[R][0]);
        d.sort_field(&grad_vel_rr_to_x_before_contact[R][0]);
        d.sort_field(&grad_vel_rr_to_q_before_contact[R][0]);
      }
    }
  }
}

Real RigidContactSolver::get_density_err_thresh() const
{
  return density_error_thresh;
}

Real RigidContactSolver::W(const Vector3r &r) const
{
  Simulation *sim = Simulation::getCurrent();
  return sim->W_with_h(r, this->supportRadius);
  //return sim->W(r);
}
Real RigidContactSolver::W_zero() const
{
  Simulation *sim = Simulation::getCurrent();
  return sim->W_zero_with_h(this->supportRadius);
  //return sim->W_zero();
}

Vector3r RigidContactSolver::gradW(const Vector3r &r) const
{
  Simulation *sim = Simulation::getCurrent();
  return sim->gradW_with_h(r, this->supportRadius);
  //return sim->gradW(r);
}

Matrix3r RigidContactSolver::gradGradW(const Vector3r &r) const
{
  Simulation *sim = Simulation::getCurrent();
  return sim->gradGradW_with_h(r, this->supportRadius);
  //return sim->gradGradW(r);
}

// ----------------------------------------------------------------------------------

Matrix3r RigidContactSolver::get_grad_gradW_to_x_rk(const Vector3r& x_rk, const Vector3r& grad_w)
{
  Matrix3r grad_to_x_rk = Matrix3r::Zero(); 
  Vector3r norm_x_kr = - x_rk / x_rk.norm();
  Simulation *sim = Simulation::getCurrent();
  if(sim->getGradientMode() == static_cast<int>(GradientMode::Complete))
    grad_to_x_rk = this->gradGradW(x_rk); 
  else 
    grad_to_x_rk = 
 -     grad_w.norm() / (x_rk).norm() * (Matrix3r::Identity() - norm_x_kr * norm_x_kr.transpose());

  return grad_to_x_rk; 
}
