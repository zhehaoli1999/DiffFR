#pragma once 
#include "SPlisHSPlasH/Common.h"
#include <vector>

using namespace std;

namespace SPH{

class RigidBodyGradientManager{
  public:
    RigidBodyGradientManager();
    void deferredInit(); 

    void step(); // perform chain rule here 
    void after_Fluid_Rigid_coupling_step(); // perform chain rule here 
    void after_Rigid_Rigid_coupling_step(); // perform chain rule here 
  
    // control the granularity of chain rule, especially in strong two-way coupling
    void perform_force_and_torque_chain_rule();  
    void perform_force_and_torque_chain_rule_Rigid();  
    void perform_velocity_chain_rule();  
    void perform_position_and_rotation_chain_rule(); 

    void copy_gradient_to_rigid_contact_solver();
    
    // interfaces to update gradient of forces and torques 
    FORCE_INLINE Matrix3r& get_grad_net_force_to_vn(const unsigned R, const unsigned RR) {return grad_net_force_to_vn[R][RR]; }
    FORCE_INLINE Matrix3r& get_grad_net_force_to_xn(const unsigned R, const unsigned RR){return grad_net_force_to_xn[R][RR]; }
    FORCE_INLINE Matrix34r& get_grad_net_force_to_qn(const unsigned R, const unsigned RR){return grad_net_force_to_qn[R][RR]; }
    FORCE_INLINE Matrix3r& get_grad_net_force_to_omega_n(const unsigned R, const unsigned RR){return grad_net_force_to_omega_n[R][RR]; }

    FORCE_INLINE Matrix3r& get_grad_net_torque_to_vn(const unsigned R, const unsigned RR){return grad_net_torque_to_vn[R][RR]; }
    FORCE_INLINE Matrix3r& get_grad_net_torque_to_xn(const unsigned R, const unsigned RR){return grad_net_torque_to_xn[R][RR]; }
    FORCE_INLINE Matrix34r& get_grad_net_torque_to_qn(const unsigned R, const unsigned RR){return grad_net_torque_to_qn[R][RR]; }
    FORCE_INLINE Matrix3r& get_grad_net_torque_to_omega_n(const unsigned R, const unsigned RR){return grad_net_torque_to_omega_n[R][RR]; }

    FORCE_INLINE Matrix3r& get_grad_vn_to_v0(const unsigned R, const unsigned RR){return grad_vn_to_v0[R][RR]; }
    FORCE_INLINE Matrix3r& get_grad_vn_to_omega0(const unsigned R, const unsigned RR){return grad_vn_to_omega0[R][RR]; }
    FORCE_INLINE Matrix3r& get_grad_omega_n_to_v0(const unsigned R, const unsigned RR){return grad_omega_n_to_v0[R][RR]; }
    FORCE_INLINE Matrix3r& get_grad_omega_n_to_omega0(const unsigned R, const unsigned RR){return grad_omega_n_to_omega0[R][RR]; }

    // interfaces to get gradient of each rigid body  
    FORCE_INLINE Matrix3r get_grad_xn_to_v0(const unsigned R, const unsigned RR) const {return grad_xn_to_v0[R][RR];}
    FORCE_INLINE Matrix3r get_grad_xn_to_omega0(const unsigned R, const unsigned RR) const {return grad_xn_to_omega0[R][RR]; }
    FORCE_INLINE Matrix43r get_grad_qn_to_v0(const unsigned R, const unsigned RR) const {return grad_qn_to_v0[R][RR];}
    FORCE_INLINE Matrix43r get_grad_qn_to_omega0(const unsigned R, const unsigned RR) const {return grad_qn_to_omega0[R][RR]; }

    // -------- helper functions ----------
    Matrix3r compute_grad_Rv_to_omega0(unsigned R, unsigned RR, Vector3r v);
    Matrix3r compute_grad_RTv_to_omega0(unsigned R, unsigned RR,Vector3r v);
    Matrix3r compute_grad_inertia_v_to_omega0(unsigned R, unsigned RR,Vector3r v);

    Matrix3r compute_grad_Rv_to_v0(unsigned R, unsigned RR,Vector3r v);
    Matrix3r compute_grad_RTv_to_v0(unsigned R, unsigned RR,Vector3r v);
    Matrix3r compute_grad_inertia_v_to_v0(unsigned R, unsigned RR,Vector3r v);

    void clear(); 
    void reset(); 

  private:
      // The overall gradient 
      vector<vector<Matrix3r>> grad_xn_to_v0;
      vector<vector<Matrix3r>> grad_xn_to_omega0;
      vector<vector<Matrix43r>> grad_qn_to_v0;
      vector<vector<Matrix43r>> grad_qn_to_omega0;
      vector<vector<Matrix3r>> grad_vn_to_v0;
      vector<vector<Matrix3r>> grad_vn_to_omega0;
      vector<vector<Matrix3r>> grad_omega_n_to_v0;
      vector<vector<Matrix3r>> grad_omega_n_to_omega0;

      // grad force of rigid body R to velocity of rigid body RR state 
      vector<vector<Matrix3r>> grad_net_force_to_vn;  
      vector<vector<Matrix3r>> grad_net_force_to_xn;  
      vector<vector<Matrix34r>> grad_net_force_to_qn;  
      vector<vector<Matrix3r>> grad_net_force_to_omega_n;  

      vector<vector<Matrix3r>> grad_net_torque_to_vn;  
      vector<vector<Matrix3r>> grad_net_torque_to_xn;  
      vector<vector<Matrix34r>> grad_net_torque_to_qn;  
      vector<vector<Matrix3r>> grad_net_torque_to_omega_n;  

      vector<vector<Matrix3r>> grad_net_force_to_v0;  
      vector<vector<Matrix3r>> grad_net_force_to_x0;  
      vector<vector<Matrix34r>> grad_net_force_to_q0;  
      vector<vector<Matrix3r>> grad_net_force_to_omega0;

      vector<vector<Matrix3r>> grad_net_torque_to_v0;  
      vector<vector<Matrix3r>> grad_net_torque_to_x0;  
      vector<vector<Matrix34r>> grad_net_torque_to_q0;  
      vector<vector<Matrix3r>> grad_net_torque_to_omega0;  
  };
} 

