#include "RigidBodyGradientManager.h"
#include "GradientUtils.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/Simulation.h"
#include "Utilities/ColorfulPrint.h"
#include "Utilities/Logger.h"

using namespace std;
using namespace SPH;

RigidBodyGradientManager::RigidBodyGradientManager()
{}

void RigidBodyGradientManager::deferredInit()
{
  auto sim = Simulation::getCurrent();
  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();

  grad_xn_to_v0.resize(n_rigid_obj); 
  grad_xn_to_omega0.resize(n_rigid_obj); 
  grad_qn_to_v0.resize(n_rigid_obj); 
  grad_qn_to_omega0.resize(n_rigid_obj); 

  grad_vn_to_v0.resize(n_rigid_obj); 
  grad_vn_to_omega0.resize(n_rigid_obj); 
  grad_omega_n_to_v0.resize(n_rigid_obj); 
  grad_omega_n_to_omega0.resize(n_rigid_obj); 

  // -----------------------------------------------------

  grad_net_force_to_vn.resize(n_rigid_obj);
  grad_net_force_to_xn.resize(n_rigid_obj);
  grad_net_force_to_qn.resize(n_rigid_obj);
  grad_net_force_to_omega_n.resize(n_rigid_obj);

  grad_net_torque_to_vn.resize(n_rigid_obj);
  grad_net_torque_to_xn.resize(n_rigid_obj);
  grad_net_torque_to_qn.resize(n_rigid_obj);
  grad_net_torque_to_omega_n.resize(n_rigid_obj);

  // -----------------------------------------------------

  grad_net_force_to_v0.resize(n_rigid_obj);
  grad_net_force_to_x0.resize(n_rigid_obj);
  grad_net_force_to_q0.resize(n_rigid_obj);
  grad_net_force_to_omega0.resize(n_rigid_obj);

  grad_net_torque_to_v0.resize(n_rigid_obj);
  grad_net_torque_to_x0.resize(n_rigid_obj);
  grad_net_torque_to_q0.resize(n_rigid_obj);
  grad_net_torque_to_omega0.resize(n_rigid_obj);

  // -----------------------------------------------------
  
  for(int R = 0; R < n_rigid_obj; R++)
  {
    grad_xn_to_v0[R].resize(n_rigid_obj); 
    grad_xn_to_omega0[R].resize(n_rigid_obj); 
    grad_qn_to_v0[R].resize(n_rigid_obj); 
    grad_qn_to_omega0[R].resize(n_rigid_obj); 

    grad_vn_to_v0[R].resize(n_rigid_obj); 
    grad_vn_to_omega0[R].resize(n_rigid_obj); 
    grad_omega_n_to_v0[R].resize(n_rigid_obj); 
    grad_omega_n_to_omega0[R].resize(n_rigid_obj); 

    // -----------------------------------------------------

    grad_net_force_to_vn[R].resize(n_rigid_obj);
    grad_net_force_to_xn[R].resize(n_rigid_obj);
    grad_net_force_to_qn[R].resize(n_rigid_obj);
    grad_net_force_to_omega_n[R].resize(n_rigid_obj);

    grad_net_torque_to_vn[R].resize(n_rigid_obj);
    grad_net_torque_to_xn[R].resize(n_rigid_obj);
    grad_net_torque_to_qn[R].resize(n_rigid_obj);
    grad_net_torque_to_omega_n[R].resize(n_rigid_obj);

    // -----------------------------------------------------

    grad_net_force_to_v0[R].resize(n_rigid_obj);
    grad_net_force_to_x0[R].resize(n_rigid_obj);
    grad_net_force_to_q0[R].resize(n_rigid_obj);
    grad_net_force_to_omega0[R].resize(n_rigid_obj);

    grad_net_torque_to_v0[R].resize(n_rigid_obj);
    grad_net_torque_to_x0[R].resize(n_rigid_obj);
    grad_net_torque_to_q0[R].resize(n_rigid_obj);
    grad_net_torque_to_omega0[R].resize(n_rigid_obj);
  }

  reset();
}

void RigidBodyGradientManager::perform_force_and_torque_chain_rule()
{
  auto sim = Simulation::getCurrent();
  GradientMode gradient_mode = static_cast<GradientMode>(sim->getGradientMode());
  RigidBodyMode rb_mode = static_cast<RigidBodyMode>(sim->getRigidBodyMode());

  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();
  // -------------------------------------------------
  for(int R = 0; R < n_rigid_obj; R++)
  {
    for(int RR = 0; RR < n_rigid_obj; RR++)
    {
      grad_net_force_to_v0[R][RR].setZero();
      grad_net_force_to_omega0[R][RR].setZero();
      grad_net_torque_to_v0[R][RR].setZero();
      grad_net_torque_to_omega0[R][RR].setZero();
    }
  }
  // -------------------------------------------------

  // First loop, accumulate all gradient of forces and torques 
  for(int R = 0; R < n_rigid_obj; R ++)
  {
    if(sim->getBoundaryModel(R)->getRigidBodyObject()->isDynamic() == false)
      continue;

    auto bm = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(R));
    auto rbo = bm->getRigidBodyObject();

    for(int RR = 0; RR < n_rigid_obj; RR++)
    {
      if(sim->getBoundaryModel(RR)->getRigidBodyObject()->isDynamic() == false)
        continue;

      for(int Rk = 0; Rk < n_rigid_obj; Rk++)
      {
        if(sim->getBoundaryModel(Rk)->getRigidBodyObject()->isDynamic() == false)
          continue;

        grad_net_force_to_v0[R][RR] += grad_net_force_to_vn[R][Rk] * grad_vn_to_v0[Rk][RR] + 
          grad_net_force_to_omega_n[R][Rk] * grad_omega_n_to_v0[Rk][RR]; 

        grad_net_torque_to_v0[R][RR] += grad_net_torque_to_vn[R][Rk] * grad_vn_to_v0[Rk][RR] + 
          grad_net_torque_to_omega_n[R][Rk] * grad_omega_n_to_v0[Rk][RR]; 

        grad_net_force_to_omega0[R][RR] += grad_net_force_to_vn[R][Rk] * grad_vn_to_omega0[Rk][RR] + 
          grad_net_force_to_omega_n[R][Rk] * grad_omega_n_to_omega0[Rk][RR]; 

        grad_net_torque_to_omega0[R][RR] += grad_net_torque_to_vn[R][Rk] * grad_vn_to_omega0[Rk][RR] + 
          grad_net_torque_to_omega_n[R][Rk] * grad_omega_n_to_omega0[Rk][RR]; 
        
        // After use, need to clear 
        //grad_net_force_to_vn[R][Rk].setZero();
        //grad_net_force_to_omega_n[R][Rk].setZero();
        //grad_net_torque_to_vn[R][Rk].setZero();
        //grad_net_torque_to_omega_n[R][Rk].setZero();
        
      }
    }
  }
}

void RigidBodyGradientManager::perform_force_and_torque_chain_rule_Rigid()
{
  auto sim = Simulation::getCurrent();
  GradientMode gradient_mode = static_cast<GradientMode>(sim->getGradientMode());
  RigidBodyMode rb_mode = static_cast<RigidBodyMode>(sim->getRigidBodyMode());

  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();
  // -------------------------------------------------
  for(int R = 0; R < n_rigid_obj; R++)
  {
    for(int RR = 0; RR < n_rigid_obj; RR++)
    {
      grad_net_force_to_v0[R][RR].setZero();
      grad_net_force_to_omega0[R][RR].setZero();
      grad_net_torque_to_v0[R][RR].setZero();
      grad_net_torque_to_omega0[R][RR].setZero();
    }
  }
  // -------------------------------------------------

  // First loop, accumulate all gradient of forces and torques 
  for(int R = 0; R < n_rigid_obj; R ++)
  {
    if(sim->getBoundaryModel(R)->getRigidBodyObject()->isDynamic() == false)
      continue;

    auto bm = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(R));
    auto rbo = bm->getRigidBodyObject();

    for(int RR = 0; RR < n_rigid_obj; RR++)
    {
      if(sim->getBoundaryModel(RR)->getRigidBodyObject()->isDynamic() == false)
        continue;

      for(int Rk = 0; Rk < n_rigid_obj; Rk++)
      {
        if(sim->getBoundaryModel(Rk)->getRigidBodyObject()->isDynamic() == false)
          continue;

        grad_net_force_to_v0[R][RR] += grad_net_force_to_vn[R][Rk] * grad_vn_to_v0[Rk][RR] + 
          grad_net_force_to_omega_n[R][Rk] * grad_omega_n_to_v0[Rk][RR] +
          grad_net_force_to_xn[R][Rk] * grad_xn_to_v0[Rk][RR];
          grad_net_force_to_qn[R][Rk] * grad_qn_to_v0[Rk][RR]; 

        grad_net_torque_to_v0[R][RR] += grad_net_torque_to_vn[R][Rk] * grad_vn_to_v0[Rk][RR] + 
          grad_net_torque_to_omega_n[R][Rk] * grad_omega_n_to_v0[Rk][RR]+
          grad_net_torque_to_xn[R][Rk] * grad_xn_to_v0[Rk][RR];
          grad_net_torque_to_qn[R][Rk] * grad_qn_to_v0[Rk][RR]; 

        if (sim->isDebug()) {
          LOG_INFO << Utilities::RedHead() <<  "[ R = " << R << ", Rk = " << Rk << ", RR = " << RR << " ]"<< Utilities::YellowTail() << "\n";
          LOG_INFO << Utilities::RedHead() <<  "grad_net_torque_to_vn[R][Rk] = " << grad_net_torque_to_vn[R][Rk]  << Utilities::YellowTail() << "\n";
          LOG_INFO << Utilities::RedHead() <<  " grad_vn_to_v0[Rk][RR] = " <<  grad_vn_to_v0[Rk][RR] << Utilities::YellowTail() << "\n";
          LOG_INFO << Utilities::RedHead() <<  " grad_net_torque_to_omega_n[R][Rk] = " << grad_net_torque_to_omega_n[R][Rk]  << Utilities::YellowTail() << "\n";
          LOG_INFO << Utilities::RedHead() <<  "grad_omega_n_to_v0[Rk][RR]= " << grad_omega_n_to_v0[Rk][RR]  << Utilities::YellowTail() << "\n";
        }

        grad_net_force_to_omega0[R][RR] += grad_net_force_to_vn[R][Rk] * grad_vn_to_omega0[Rk][RR] + 
          grad_net_force_to_omega_n[R][Rk] * grad_omega_n_to_omega0[Rk][RR]+
          grad_net_force_to_xn[R][Rk] * grad_xn_to_omega0[Rk][RR] +
          grad_net_force_to_qn[R][Rk] * grad_qn_to_omega0[Rk][RR]; 

        grad_net_torque_to_omega0[R][RR] += grad_net_torque_to_vn[R][Rk] * grad_vn_to_omega0[Rk][RR] + 
          grad_net_torque_to_omega_n[R][Rk] * grad_omega_n_to_omega0[Rk][RR] +
          grad_net_torque_to_xn[R][Rk] * grad_xn_to_omega0[Rk][RR] +
          grad_net_torque_to_qn[R][Rk] * grad_qn_to_omega0[Rk][RR]; 

        
        // After use, need to clear 
        //grad_net_force_to_vn[R][Rk].setZero();
        //grad_net_force_to_omega_n[R][Rk].setZero();
        //grad_net_force_to_xn[R][Rk].setZero();
        //grad_net_force_to_qn[R][Rk].setZero();

        //grad_net_torque_to_vn[R][Rk].setZero();
        //grad_net_torque_to_omega_n[R][Rk].setZero();
        //grad_net_torque_to_xn[R][Rk].setZero();
        //grad_net_torque_to_qn[R][Rk].setZero();
      }
    }
  }
}

// -------------------------------------------------------------------------------------
Matrix3r RigidBodyGradientManager::compute_grad_Rv_to_omega0(unsigned R, unsigned RR, Vector3r v)
{
  auto rbo = Simulation::getCurrent()->getBoundaryModel(R)->getRigidBodyObject();
  Matrix34r grad_Rv_to_q = get_grad_Rqp_to_q(rbo->getRotation(), v);
  return grad_Rv_to_q * grad_qn_to_omega0[R][RR];
}

Matrix3r RigidBodyGradientManager::compute_grad_RTv_to_omega0(unsigned R, unsigned RR, Vector3r v)
{
  auto rbo = Simulation::getCurrent()->getBoundaryModel(R)->getRigidBodyObject();
  Matrix34r grad_RTv_to_q = get_grad_RqTp_to_q(rbo->getRotation(), v);
  return grad_RTv_to_q * grad_qn_to_omega0[R][RR];
}

Matrix3r RigidBodyGradientManager::compute_grad_Rv_to_v0(unsigned R, unsigned RR,Vector3r v)
{
  auto rbo = Simulation::getCurrent()->getBoundaryModel(R)->getRigidBodyObject();
  Matrix34r grad_Rv_to_q = get_grad_Rqp_to_q(rbo->getRotation(), v);
  return grad_Rv_to_q * grad_qn_to_v0[R][RR];
}

Matrix3r RigidBodyGradientManager::compute_grad_RTv_to_v0(unsigned R, unsigned RR,Vector3r v)
{
  auto rbo = Simulation::getCurrent()->getBoundaryModel(R)->getRigidBodyObject();
  Matrix34r grad_RTv_to_q = get_grad_RqTp_to_q(rbo->getRotation(), v);
  return grad_RTv_to_q * grad_qn_to_v0[R][RR];
}

// -------------------------------------------------------------------------------------

Matrix3r RigidBodyGradientManager::compute_grad_inertia_v_to_omega0(unsigned R, unsigned RR,Vector3r v)
{
  auto rbo = Simulation::getCurrent()->getBoundaryModel(R)->getRigidBodyObject();
  Matrix3r inertia_0 = rbo->getInertiaTensor0();
  Matrix3r rotation = rbo->getRotation().toRotationMatrix();
  return ( compute_grad_Rv_to_omega0(R, RR, inertia_0 * rotation.transpose() * v)
  +  rotation *inertia_0* compute_grad_RTv_to_omega0(R, RR, v)  );
}

Matrix3r RigidBodyGradientManager::compute_grad_inertia_v_to_v0(unsigned R, unsigned RR,Vector3r v)
{
  auto rbo = Simulation::getCurrent()->getBoundaryModel(R)->getRigidBodyObject();
  Matrix3r inertia_0 = rbo->getInertiaTensor0();
  Matrix3r rotation = rbo->getRotation().toRotationMatrix();
  return ( compute_grad_Rv_to_v0(R, RR, inertia_0 * rotation.transpose() * v)
  +  rotation *inertia_0* compute_grad_RTv_to_v0(R, RR, v)  );
}

// -------------------------------------------------------------------------------------

void RigidBodyGradientManager::perform_velocity_chain_rule()
{
  auto sim = Simulation::getCurrent();
  GradientMode gradient_mode = static_cast<GradientMode>(sim->getGradientMode());
  RigidBodyMode rb_mode = static_cast<RigidBodyMode>(sim->getRigidBodyMode());

  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();
  // FIXME: One detail: dt may change, need to pass the old value of timestep
  Real dt = TimeManager::getCurrent()->getTimeStepSize();
  
  for(int R = 0; R < n_rigid_obj; R++)
  {
    if(sim->getBoundaryModel(R)->getRigidBodyObject()->isDynamic() == false)
      continue;

    auto rbo = sim->getBoundaryModel(R)->getRigidBodyObject(); 
    
    Matrix3r inertia = rbo->getInertiaTensorWorld();
    Matrix3r inv_inertia = rbo->getInertiaTensorInverseWorld();
    auto omega = rbo->getAngularVelocity();
    Vector3r temp_v = Vector3r::Zero();
    Vector3r force, torque;
    sim->getBoundaryModel(R)->getForceAndTorque(force, torque);

    for(int RR = 0; RR < n_rigid_obj; RR++)
    {
      if(sim->getBoundaryModel(RR)->getRigidBodyObject()->isDynamic() == false)
        continue;

      grad_vn_to_v0[R][RR] = grad_vn_to_v0[R][RR] + dt * rbo->getInvMass() * grad_net_force_to_v0[R][RR]; 
      grad_vn_to_omega0[R][RR] = grad_vn_to_omega0[R][RR] + dt * rbo->getInvMass() * grad_net_force_to_omega0[R][RR]; 

      Matrix3r grad_Tau_to_omega0= Matrix3r::Zero();
      Matrix3r grad_Tau_to_v0 = Matrix3r::Zero();
      if(rb_mode == RigidBodyMode::WithGyroscopic)
      {
        temp_v = (inertia * omega).cross(omega) + torque; // TODO: duplicate computation here.

        Matrix3r grad_L_cross_omega_to_omega0 = skewMatrix(inertia * omega) * grad_omega_n_to_omega0[R][RR]
          + skewMatrix(omega).transpose() * (inertia * grad_omega_n_to_omega0[R][RR] + compute_grad_inertia_v_to_omega0(R, RR, omega) );

        Matrix3r grad_L_cross_omega_to_v0 = skewMatrix(inertia * omega) * grad_omega_n_to_v0[R][RR]
          + skewMatrix(omega).transpose() * (inertia * grad_omega_n_to_v0[R][RR] + compute_grad_inertia_v_to_v0(R, RR, omega) );

        if(sim->isDebug())
        {
          LOG_INFO << Utilities::CyanHead() <<  "grad_L_cross_omega_to_omega0 = " << grad_L_cross_omega_to_omega0 << Utilities::YellowTail() << "\n";
          LOG_INFO << Utilities::CyanHead() <<  "grad_L_cross_omega_to_v0 = " << grad_L_cross_omega_to_v0 << Utilities::YellowTail() << "\n";
        }

        grad_Tau_to_omega0 = (grad_L_cross_omega_to_omega0 + grad_net_torque_to_omega0[R][RR]);
        grad_Tau_to_v0 = (grad_L_cross_omega_to_v0 + grad_net_torque_to_v0[R][RR]);
      }
      else
      {
        temp_v = torque; // TODO: duplicate computation here.
        grad_Tau_to_omega0 = (grad_net_torque_to_omega0[R][RR]);
        grad_Tau_to_v0 = (grad_net_torque_to_v0[R][RR]);
      }
      Matrix3r grad_inv_inertia_v_to_omega0 = inv_inertia * compute_grad_inertia_v_to_omega0(R, RR, inv_inertia * temp_v);
      Matrix3r grad_inv_inertia_v_to_v0 = inv_inertia * compute_grad_inertia_v_to_v0(R, RR, inv_inertia * temp_v);

      grad_omega_n_to_omega0[R][RR] = grad_omega_n_to_omega0[R][RR] + dt * (grad_inv_inertia_v_to_omega0
        + rbo->getInertiaTensorInverseWorld() * grad_Tau_to_omega0);
      grad_omega_n_to_v0[R][RR] = grad_omega_n_to_v0[R][RR] + dt * (grad_inv_inertia_v_to_v0
        + rbo->getInertiaTensorInverseWorld() * grad_Tau_to_v0);

      // ------------------------------------------------------------
      if(sim->isDebug())
      {
        //LOG_INFO << Utilities::YellowHead() <<  "net force = " << force.transpose() << Utilities::YellowTail() << "\n";
        //LOG_INFO << Utilities::YellowHead() <<  "net torque = " << torque.transpose() << Utilities::YellowTail() << "\n";
        //LOG_INFO << Utilities::CyanHead() <<  "dt = " << dt << Utilities::YellowTail() << "\n";
        //LOG_INFO << Utilities::CyanHead() <<  "inv_mass = " << rbo->getInvMass() << Utilities::YellowTail() << "\n";
        //LOG_INFO << Utilities::CyanHead() <<  "omega = " << omega << Utilities::YellowTail() << "\n";
        LOG_INFO << Utilities::CyanHead() <<  "[ R = " << R << ", RR = "<< RR << " ] "<< Utilities::YellowTail() << "\n";

        LOG_INFO << Utilities::CyanHead() <<  "grad_inv_inertia_v_to_omega0 = " << grad_inv_inertia_v_to_omega0 << Utilities::YellowTail() << "\n";
        LOG_INFO << Utilities::CyanHead() <<  "grad_inv_inertia_v_to_v0 = " << grad_inv_inertia_v_to_v0 << Utilities::YellowTail() << "\n";

        LOG_INFO << Utilities::CyanHead() <<  "grad_Tau_to_v0 = " << grad_Tau_to_v0 << Utilities::YellowTail() << "\n";
        LOG_INFO << Utilities::CyanHead() <<  "grad_Tau_to_omega0 = " << grad_Tau_to_omega0 << Utilities::YellowTail() << "\n";
        LOG_INFO << Utilities::CyanHead() <<  "grad_Tau_to_v0 = " << grad_Tau_to_v0 << Utilities::YellowTail() << "\n";

        LOG_INFO << Utilities::CyanHead() <<  "grad_net_force_to_vn = \n" << grad_net_force_to_vn[R][RR]<< Utilities::YellowTail() << "\n";
        LOG_INFO << Utilities::CyanHead() <<  "grad_net_force_to_omega_n = \n" << grad_net_force_to_omega_n[R][RR] << Utilities::YellowTail() << "\n";
        LOG_INFO << Utilities::CyanHead() <<  "grad_net_torque_to_vn = \n" << grad_net_torque_to_vn[R][RR] << Utilities::YellowTail() << "\n";
        LOG_INFO << Utilities::CyanHead() <<  "grad_net_torque_to_omega_n = \n" << grad_net_torque_to_omega_n[R][RR] << Utilities::YellowTail() << "\n";
        ////
        LOG_INFO << Utilities::CyanHead() <<  "grad_v_to_v0 = \n" << grad_vn_to_v0[R][RR] << Utilities::YellowTail() << "\n";
        LOG_INFO << Utilities::CyanHead() <<  "grad_v_to_omega0 = \n" << grad_vn_to_omega0[R][RR] << Utilities::YellowTail() << "\n";
        LOG_INFO << Utilities::CyanHead() <<  "grad_omega_to_omega0 = \n" << grad_omega_n_to_omega0[R][RR] << Utilities::YellowTail() << "\n";
        LOG_INFO << Utilities::CyanHead() <<  "grad_omega_to_v0 = \n" << grad_omega_n_to_v0[R][RR] << Utilities::YellowTail() << "\n";

        LOG_INFO << Utilities::CyanHead() <<  "grad_net_torque_to_v0 = \n" << grad_net_torque_to_v0[R][RR] << Utilities::YellowTail() << "\n";
        LOG_INFO << "=================================== \n" ; 
      }
    }

  }

}

void RigidBodyGradientManager::perform_position_and_rotation_chain_rule()
{
  Simulation *sim = Simulation::getCurrent();
  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();
  Real dt = TimeManager::getCurrent()->getTimeStepSize();

  for(int R = 0; R < n_rigid_obj; R ++)
  {
      if(sim->getBoundaryModel(R)->getRigidBodyObject()->isDynamic() == false)
        continue;
      auto bm = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(R));
      auto rbo = bm->getRigidBodyObject();

      Quaternionr q = rbo->getRotation();
      Vector3r omega = rbo->getAngularVelocity();

      Vector3r new_omega = omega; // Here we already update the omega

      auto p = Quaternionr(0., new_omega[0], new_omega[1], new_omega[2]);
      Quaternionr new_q = q;
      new_q.coeffs() += dt * 0.5 * (p * new_q).coeffs();
      auto new_qn = new_q.normalized();
      Vector4r new_qnv = Vector4r(new_qn.w(), new_qn.x(), new_qn.y(), new_qn.z());

      Matrix4r grad_p_q_product_to_q = get_grad_p_q_product_to_q(p);
      Matrix43r grad_p_q_product_to_omega = get_grad_omega_q_product_to_omega(q);

      Matrix4r grad_normalized_q_to_q =(Matrix4r::Identity() - new_qnv * new_qnv.transpose()) / new_q.norm();

      for(int RR = 0; RR < n_rigid_obj; RR++)
      {
          if(sim->getBoundaryModel(RR)->getRigidBodyObject()->isDynamic() == false)
            continue;
          grad_xn_to_v0[R][RR] += dt * grad_vn_to_v0[R][RR];
          grad_xn_to_omega0[R][RR] += dt * grad_vn_to_omega0[R][RR];
          // -------------------------------------------------------------

          grad_qn_to_v0[R][RR] = grad_normalized_q_to_q * (grad_qn_to_v0[R][RR] +
            dt / 2. * (grad_p_q_product_to_omega * grad_omega_n_to_v0[R][RR] +
            grad_p_q_product_to_q * grad_qn_to_v0[R][RR] )
          );
          grad_qn_to_omega0[R][RR] = grad_normalized_q_to_q * (grad_qn_to_omega0[R][RR] +
            dt / 2. * (grad_p_q_product_to_omega * grad_omega_n_to_omega0[R][RR] +
            grad_p_q_product_to_q * grad_qn_to_omega0[R][RR] )
          );

        // ----------------------------------------------------------
        if(sim->isDebug())
        {
          LOG_INFO << Utilities::YellowHead() <<  "[ R = " << R << ", RR = "<< RR << " ] "<< Utilities::YellowTail() << "\n";
          LOG_INFO << Utilities::YellowHead() <<  "grad_x_to_v0 = \n" << grad_xn_to_v0[R][RR] << Utilities::YellowTail() << "\n";
          LOG_INFO << Utilities::YellowHead() <<  "grad_x_to_omega0 = \n" << grad_xn_to_omega0[R][RR] << Utilities::YellowTail() << "\n";
          LOG_INFO << Utilities::YellowHead() <<  "grad_q_to_v0 = \n" << grad_qn_to_v0[R][RR] << Utilities::YellowTail() << "\n";
          LOG_INFO << Utilities::YellowHead() <<  "grad_q_to_omega0 = \n" << grad_qn_to_omega0[R][RR] << Utilities::YellowTail() << "\n";
        }
      }
    }

}

void RigidBodyGradientManager::after_Fluid_Rigid_coupling_step()
{
  perform_force_and_torque_chain_rule(); 
  perform_velocity_chain_rule(); 
}
void RigidBodyGradientManager::after_Rigid_Rigid_coupling_step()
{
  if(Simulation::getCurrent()->useRigidContactSolver())
  {
    perform_force_and_torque_chain_rule_Rigid(); 
    perform_velocity_chain_rule(); 
  }
  perform_position_and_rotation_chain_rule();
}

void RigidBodyGradientManager::step()
{
  perform_force_and_torque_chain_rule(); 
  perform_velocity_chain_rule(); 
  perform_position_and_rotation_chain_rule();
}

void RigidBodyGradientManager::reset()
{
  auto sim = Simulation::getCurrent();
  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();

  for(int R = 0; R < n_rigid_obj; R++)
  {
    for(int RR = 0; RR < n_rigid_obj; RR++)
    {
      grad_xn_to_v0[R][RR].setZero(); 
      grad_xn_to_omega0[R][RR].setZero(); 
      grad_qn_to_omega0[R][RR].setZero(); 
      grad_qn_to_v0[R][RR].setZero(); 
      grad_vn_to_omega0[R][RR].setZero(); 
      grad_omega_n_to_v0[R][RR].setZero(); 
      if(RR == R)
      {
        grad_vn_to_v0[R][RR].setIdentity();
        grad_omega_n_to_omega0[R][RR].setIdentity();
      }
      else {
        grad_vn_to_v0[R][RR].setZero();
        grad_omega_n_to_omega0[R][RR].setZero();
      }
      grad_net_force_to_vn[R][RR].setZero(); 
      grad_net_force_to_xn[R][RR].setZero(); 
      grad_net_force_to_qn[R][RR].setZero(); 
      grad_net_force_to_omega_n[R][RR].setZero(); 

      grad_net_torque_to_vn[R][RR].setZero(); 
      grad_net_torque_to_xn[R][RR].setZero(); 
      grad_net_torque_to_qn[R][RR].setZero(); 
      grad_net_torque_to_omega_n[R][RR].setZero(); 

      // ------------------------------------------------

      grad_net_force_to_v0[R][RR].setZero(); 
      grad_net_force_to_x0[R][RR].setZero(); 
      grad_net_force_to_q0[R][RR].setZero(); 
      grad_net_force_to_omega0[R][RR].setZero(); 

      grad_net_torque_to_v0[R][RR].setZero(); 
      grad_net_torque_to_x0[R][RR].setZero(); 
      grad_net_torque_to_q0[R][RR].setZero(); 
      grad_net_torque_to_omega0[R][RR].setZero(); 
    }
  }
}

void RigidBodyGradientManager::copy_gradient_to_rigid_contact_solver()
{
  auto sim = Simulation::getCurrent();
  const unsigned n_rigid_obj = sim->numberOfBoundaryModels();
  auto rigid_contact_solver = sim->getSimulatorBase()->getBoundarySimulator()->getRigidContactSolver();

  for(int R = 0; R < n_rigid_obj; R++)
  {
    for(int RR = 0; RR < n_rigid_obj; RR++)
    {
      

    }
  }
}
