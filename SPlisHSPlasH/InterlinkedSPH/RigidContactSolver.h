#pragma once 
#include "SPlisHSPlasH/Common.h"

#include <algorithm>
#include <utility>
#include <vector>

#define PRAGMA_OMP_FOR _Pragma("omp parallel for") 
#define PRAGMA_OMP_ATOMIC _Pragma("omp atomic") 
//#define PRAGMA_OMP_FOR_REDUCE(arg, ...) _Pragma(std::string("omp parallel for reduction(+:")+std::string(arg))

#define forall_rigid_body_objects(code) \
for(int R = 0; R < n_rigid_obj; R ++) \
{\
  auto bm = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(R)); \
  auto rbo = bm->getRigidBodyObject(); \
  auto n_particles = bm->numberOfParticles();\
  auto vel_R = rbo->getVelocity(); \
  auto omega_R = rbo->getAngularVelocity(); \
  auto mass_R = rbo->getMass();\
  auto inertia_R = rbo->getInertiaTensorWorld(); \
  code \
}

#define forall_rigid_particles(code) \
for (int i = 0; i < n_total_rigid_particles; i++)\
{\
  auto index = m_particle_indices[i];\
  unsigned R = index.first;\
  unsigned r = index.second;\
  \
  auto bm = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(R)); \
  auto rbo = bm->getRigidBodyObject(); \
  auto n_particles = bm->numberOfParticles();\
  auto vel_R = rbo->getVelocity(); \
  auto omega_R = rbo->getAngularVelocity(); \
  auto mass_R = rbo->getMass();\
  auto inertia_R = rbo->getInertiaTensorWorld(); \
  \
  auto x_r = bm->getPosition(r);  \
  auto r_r = x_r - rbo->getPosition(); \
  auto r_r0 = bm->getPosition0(r); \
  code \
}

#define forall_rigid_particles_in_contact(code) \
for (int i = 0; i < m_particle_indices_in_contact.size(); i++)\
{\
  auto index = m_particle_indices_in_contact[i];\
  unsigned R = index.first;\
  unsigned r = index.second;\
  \
  auto bm = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModel(R)); \
  auto rbo = bm->getRigidBodyObject(); \
  auto n_particles = bm->numberOfParticles();\
  auto vel_R = rbo->getVelocity(); \
  auto omega_R = rbo->getAngularVelocity(); \
  auto mass_R = rbo->getMass();\
  auto inertia_R = rbo->getInertiaTensorWorld(); \
  \
  auto x_r = bm->getPosition(r);  \
  auto r_r = x_r - rbo->getPosition(); \
  auto r_r0 = bm->getPosition0(r); \
  code \
}

/** Loop over the rigid neighbors of a rigid particle.
* Simulation *sim and unsigned int rigidbodyIndex R must be defined.
*/
#define forall_rigid_neighbors(code) \
for (unsigned int pid = sim->numberOfFluidModels(); pid < sim->numberOfPointSets(); pid++) \
{ \
	BoundaryModel_Akinci2012 *bm_neighbor = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModelFromPointSet(pid)); \
  for (unsigned int kk = 0; kk < sim->numberOfNeighbors(sim->numberOfFluidModels()+R, pid, r); kk++) \
  { \
    const unsigned int k = sim->getNeighbor(sim->numberOfFluidModels()+R, pid, r, kk); \
    const unsigned R_k = pid - sim->numberOfFluidModels(); \
    const Vector3r &x_k = bm_neighbor->getPosition(k); \
    auto rbo_k = bm_neighbor->getRigidBodyObject(); \
    const Vector3r r_k = x_k - rbo_k->getPosition(); \
    const Vector3r r_k0 = bm_neighbor->getPosition0(k);\
    \
    code \
    \
	} \
}


using namespace std; 

namespace SPH{

  // This class implements the SPH solver for rigid body contact in paper 
  // "Interlinked SPH Pressure Solvers for Strong Fluid-Rigid Coupling"
  class RigidContactSolver // : public GenParam::ParameterObject
  {
  public:
    RigidContactSolver(); 
    
    void solveRigidContact();
    Real solveRigidContactIteration(const Real density_error_thresh);
    void solveRigidContactPenalty(); 
    void solveFriction();
    
    void reset();
    
    //void initParameters(); 
    bool is_in_contact(const unsigned R, const unsigned r) const; 
    
    void clearGradientBeforeSolver(); 
    void clearGradientDuringIteration(); 

    void update_rigid_body_gradient_manager(); 
    
    void performNeighborhoodSearchSort();

    // compute the density, volume, velocity of each rigid particle
    void beforeIterationInitialize(); 
    void beforePenaltyInitialize(); 
    
    // Return avg_density_err
    Real step();   

    Real get_density_err_thresh() const;
    
    Real W(const Vector3r& r) const; 
    Real W_zero() const; 
    Vector3r gradW(const Vector3r& r) const; 
    Matrix3r gradGradW(const Vector3r& r) const; 

    Matrix3r get_grad_gradW_to_x_rk(const Vector3r& x_rk, const Vector3r& gradW); 

    //static int FRICTION_COEFF; 
  private:
  
    vector<vector<Real>> pressure;  // artificial pressure of rigid particles 
    vector<vector<Vector3r>> nabla_pressure;  // artificial pressure of rigid particles 
    vector<vector<Real>> density;  // density of rigid particles 
    vector<vector<Real>> density0;  // initial density of rigid particles 
    vector<vector<Vector3r>> vel; 
    vector<vector<Vector3r>> vel_rr; 
    vector<vector<Real>> vol; 
    vector<vector<Real>> vol0; 
    vector<vector<Real>> s; 
    vector<vector<Vector3r>> b_from_v_rr_r; 
    vector<vector<Real>> b; 
    vector<vector<Real>> div_vel_rr; 
    vector<vector<Real>> div_v_s; 
    //Real vol0; // volume 
    unsigned n_total_rigid_particles; 
    unsigned num_contacts; 
    Real friction_coeff; 
    bool use_friction; 
    Real density_error_thresh; 
    Real supportRadius; 

    bool use_penalty_force; 

    vector<Vector3r> vel_rr_R;
    vector<Vector3r> omega_rr_R;
    vector<std::pair<unsigned, unsigned>> m_particle_indices ;
    vector<std::pair<unsigned, unsigned>> m_particle_indices_in_contact;

    // The gradient of rigid particle r of rigid body R to rigid body R_k 
    vector<vector<vector<Vector3r>>> grad_pressure_to_vel_before_contact; 
    vector<vector<vector<Vector3r>>> grad_pressure_to_omega_before_contact; 
    vector<vector<vector<Vector3r>>> grad_pressure_to_x_before_contact; 
    vector<vector<vector<Vector4r>>> grad_pressure_to_q_before_contact; 

    vector<vector<vector<Vector3r>>> grad_s_to_vel_before_contact; 
    vector<vector<vector<Vector3r>>> grad_s_to_omega_before_contact; 
    vector<vector<vector<Vector3r>>> grad_s_to_x_before_contact; 
    vector<vector<vector<Vector4r>>> grad_s_to_q_before_contact; 

    vector<vector<vector<Vector3r>>> grad_div_vel_rr_to_vel_before_contact; 
    vector<vector<vector<Vector3r>>> grad_div_vel_rr_to_omega_before_contact; 
    vector<vector<vector<Vector3r>>> grad_div_vel_rr_to_x_before_contact; 
    vector<vector<vector<Vector4r>>> grad_div_vel_rr_to_q_before_contact; 

    vector<vector<vector<Matrix3r>>> grad_nabla_p_r_to_vel_before_contact; 
    vector<vector<vector<Matrix3r>>> grad_nabla_p_r_to_omega_before_contact; 
    vector<vector<vector<Matrix3r>>> grad_nabla_p_r_to_x_before_contact; 
    vector<vector<vector<Matrix34r>>> grad_nabla_p_r_to_q_before_contact; 

    // The gradient of rigid particle r of rigid body R to rigid body R_k 
    vector<vector<vector<Matrix3r>>> grad_vel_rr_to_vel_before_contact; 
    vector<vector<vector<Matrix3r>>> grad_vel_rr_to_omega_before_contact; 
    vector<vector<vector<Matrix3r>>> grad_vel_rr_to_x_before_contact; 
    vector<vector<vector<Matrix34r>>> grad_vel_rr_to_q_before_contact; 

    //------------------------------------------------------

    // gradient from friction (for now we can ignore this)
    vector<vector<Matrix3r>> grad_friction_to_vel_before_contact;
    vector<vector<Matrix3r>> grad_friction_to_omega_before_contact;
    vector<vector<Matrix3r>> grad_friction_to_x_before_contact;
    vector<vector<Matrix34r>> grad_friction_to_q_before_contact;

    // The gradient of forces 
    // The gradient of force / torque of rigid body R to rigid body R_k 
    vector<vector<Matrix3r>> grad_net_force_to_vel_before_contact;
    vector<vector<Matrix3r>> grad_net_force_to_omega_before_contact;
    vector<vector<Matrix3r>> grad_net_force_to_x_before_contact;
    vector<vector<Matrix34r>> grad_net_force_to_q_before_contact;

    vector<vector<Matrix3r>> grad_net_torque_to_vel_before_contact;
    vector<vector<Matrix3r>> grad_net_torque_to_omega_before_contact;
    vector<vector<Matrix3r>> grad_net_torque_to_x_before_contact;
    vector<vector<Matrix34r>> grad_net_torque_to_q_before_contact;

    // 
    vector<vector<Matrix3r>> grad_vel_rr_R_to_vel_before_contact; 
    vector<vector<Matrix3r>> grad_vel_rr_R_to_omega_before_contact; 
    vector<vector<Matrix3r>> grad_vel_rr_R_to_x_before_contact; 
    vector<vector<Matrix34r>> grad_vel_rr_R_to_q_before_contact; 
    // 
    vector<vector<Matrix3r>> grad_omega_rr_R_to_vel_before_contact; 
    vector<vector<Matrix3r>> grad_omega_rr_R_to_omega_before_contact; 
    vector<vector<Matrix3r>> grad_omega_rr_R_to_x_before_contact; 
    vector<vector<Matrix34r>> grad_omega_rr_R_to_q_before_contact; 
    
  };

}
