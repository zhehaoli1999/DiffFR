#ifndef __ArticulatedDynamicSystem_h__
#define __ArticulatedDynamicSystem_h__

#include "SPlisHSPlasH/Common.h"
#include <cassert>
#include <vector>
#include <map>

namespace SPH
{
  class RigidBodyObject;
  class BoundaryModel_Akinci2012;

  class ArticulatedDynamicSystemBase
  {
  public:
      struct Balljoint
      {
        // rigid body indices
        unsigned int index1;
        unsigned int index2;

        // joint point relative positions to two rigid body centers
        Vector3r r_body1_to_joint;
        Vector3r r_body2_to_joint;

        bool isActuator;
      };

      std::vector<RigidBodyObject *> bodies;
      std::vector<Balljoint> joints; 
      Real inv_stiffness; 

      ArticulatedDynamicSystemBase();

      virtual void init();
      virtual void timeStep();
      virtual void updateMassCenterPosition();
      virtual void reset();
      virtual std::vector<Vector3r> getJointPoints();
        
      RigidBodyObject* getRootRigidbody() { assert(bodies.size() >= 1); return bodies[0]; }


      virtual void addRigidBody(RigidBodyObject *rb){bodies.push_back(rb);}
      virtual void addJoint(unsigned index1, unsigned index2, const Vector3r &position, bool isActuator);
      virtual void addForce(const Vector3r &f, RigidBodyObject* rbo);
      virtual void addTorque(const Vector3r &t, RigidBodyObject* rbo);
      virtual void clearForceAndTorque();
      virtual const Vector3r &getPosition() const {return massCenterPosition;}
      
      virtual void checkJointsStatus();

      virtual void setAngularVelocityToJoint(RigidBodyObject* body, const Vector3r& omega);
      virtual void addAngularVelocityToJoint(RigidBodyObject* body, const Vector3r& delta_omega);

      void getJointInfoByBody(RigidBodyObject *body, RigidBodyObject *&otherBody, Vector3r &l1, Vector3r &l2);

      void setRigidInfo(RigidBodyObject *rbo, unsigned int index, BoundaryModel_Akinci2012 *bm) { rigidBodyInfo.insert({rbo, {index, bm}}); }
      void setLastRigidBoundaryModel(BoundaryModel_Akinci2012 *bm) { setRigidInfo(bodies[bodies.size() - 1], bodies.size() - 1, bm); }

      // assume bm1 must be in the system, while bm2 not
      //Matrix3r get_grad_v_to_v0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2);
      //Matrix3r get_grad_omega_to_v0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2);
      //Matrix3r get_grad_x_to_v0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2);
      Matrix43r get_grad_quaternion_to_v0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2);

      //Matrix3r get_grad_v_to_omega0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2);
      //Matrix3r get_grad_omega_to_omega0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2);
      Matrix3r get_grad_x_to_omega0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2);
      Matrix3r get_grad_x_to_v0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2);
      //Matrix43r get_grad_quaternion_to_omega0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2);

      //void get_bm_grad_information(BoundaryModel_Akinci2012 *bm, Matrix3r &grad_v_to_v0, Matrix3r &grad_v_to_omega0, Matrix3r &grad_omega_to_v0, Matrix3r &grad_omega_to_omega0,
                                   //Matrix3r &grad_x_to_v0, Matrix3r &grad_x_to_omega0, Matrix34r &grad_quaternion_to_v0, Matrix34r &grad_quaternion_to_omega0);
      
      Matrix43r compute_grad_qn_to_v(unsigned parentIdx, unsigned childIdx, Real dt);

      
      //MatrixXr get_grad_rootRB_xn_to_actuator_omega0();
      unsigned int n_actuator_joint; 
      unsigned int n_dof_actuator; // 3 * n_actuator_joint
      
    protected:
      bool findRigidIndex(RigidBodyObject *rbo, unsigned int &index);
      bool findBoundaryModel(RigidBodyObject *rbo, BoundaryModel_Akinci2012 *&bm);
      unsigned int getRigidIndex(RigidBodyObject *rbo);
      BoundaryModel_Akinci2012 *getBoundaryModel(RigidBodyObject *rbo);
      std::vector<Vector3r> forces;
      std::vector<Vector3r> torques;
      MatrixXr massMatrix;
      Vector3r massCenterPosition;

      std::map<RigidBodyObject *, std::pair<unsigned int, BoundaryModel_Akinci2012 *>> rigidBodyInfo;
      
      void timeStepLagrangian();
      void timeStepImpulseBased();

      // TODO: maybe something like "#ifdef backward" can be added
    public:
      void perform_chain_rule(const Real dt);

    protected:
      //MatrixXr m_grad_Vn_to_Vn_1;
      //MatrixXr m_grad_Vn_to_Fn_1;

      //MatrixXr m_grad_Vn_to_V0;
      //MatrixXr m_grad_Xn_to_V0;
      
      std::vector<std::vector<Matrix3r>> m_grad_vn_to_v0;
      std::vector<std::vector<Matrix3r>> m_grad_vn_to_omega0;
      std::vector<std::vector<Matrix3r>> m_grad_omega_n_to_omega0;
      std::vector<std::vector<Matrix3r>> m_grad_omega_n_to_v0;

      std::vector<std::vector<Matrix3r>> m_grad_xn_to_v0;
      std::vector<std::vector<Matrix43r>> m_grad_qn_to_v0;
      std::vector<std::vector<Matrix3r>> m_grad_xn_to_omega0;
      std::vector<std::vector<Matrix43r>> m_grad_qn_to_omega0;

      //MatrixXr m_grad_rootRB_vn_to_actuator_omega0; 
      //MatrixXr m_grad_rootRB_qn_to_actuator_omega0; 
      //MatrixXr m_grad_rootRB_omega_n_to_actuator_omega0; 
      //MatrixXr m_grad_rootRB_xn_to_actuator_omega0;

      bool m_useLagrangian;

      bool isChainRulePerformed;
  };
}

#endif
