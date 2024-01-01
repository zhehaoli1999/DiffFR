#ifndef __BoundaryModel_Akinci2012_h__
#define __BoundaryModel_Akinci2012_h__

#include "Common.h"
#include <vector>

#include "BoundaryModel.h"
#include "SPHKernels.h"

#define BACKWARD

namespace SPH 
{	

	class TimeStep;

	/** \brief The boundary model stores the information required for boundary handling
	* using the approach of Akinci et al. 2012 [AIA+12].
	*
	* References:
	* - [AIA+12] Nadir Akinci, Markus Ihmsen, Gizem Akinci, Barbara Solenthaler, and Matthias Teschner. Versatile rigid-fluid coupling for incompressible SPH. ACM Trans. Graph., 31(4):62:1-62:8, July 2012. URL: http://doi.acm.org/10.1145/2185520.2185558
	*/
	class BoundaryModel_Akinci2012 : public BoundaryModel
	{
		public:
			BoundaryModel_Akinci2012();
			virtual ~BoundaryModel_Akinci2012();

		protected:
			bool m_sorted;
			unsigned int m_pointSetIndex;

			// values required for Akinci 2012 boundary handling
			std::vector<Vector3r> m_x0;
			std::vector<Vector3r> m_x;
			std::vector<Vector3r> m_v;
			std::vector<Real> m_V;
			std::vector<Real> m_pressure; // For Band18 MLS 

#ifdef BACKWARD
      // Particle gradients 
      std::vector<Matrix3r> m_grad_force_to_v; 
      std::vector<Matrix3r> m_grad_force_to_x; 
      std::vector<Matrix34r> m_grad_force_to_quaternion; 
      std::vector<Matrix3r> m_grad_force_to_omega;

      std::vector<Matrix3r> m_grad_torque_to_omega; 
      std::vector<Matrix34r> m_grad_torque_to_quaternion; 
      std::vector<Matrix3r> m_grad_torque_to_x; 
      std::vector<Matrix3r> m_grad_torque_to_v; 

     // Sum of particle gradient to get total gradient of the whole rigidbody 
      Matrix3r grad_net_force_to_vn;
      Matrix3r grad_net_force_to_xn;
      Matrix34r grad_net_force_to_qn;
      Matrix3r grad_net_force_to_omega_n;

      Matrix3r grad_net_torque_to_omega_n; 
      Matrix3r grad_net_torque_to_vn; 
      Matrix3r grad_net_torque_to_xn; 
      Matrix34r grad_net_torque_to_qn; 

      // Final gradient of the whole rigidbody
      Matrix3r m_grad_x_to_v0; // grad x_n to v0
      Matrix43r m_grad_quaternion_to_omega0; // grad quaternion_n to omega0
      Matrix43r m_partial_grad_qn_to_omega_n; // grad quaternion_n to omega0
      Matrix3r m_grad_angle_in_radian_to_omega0; 

      Matrix3r m_grad_v_to_v0; // grad v_n to v0
      Matrix3r m_grad_omega_to_omega0; // grad omega_n to omega0
      
      Matrix3r m_grad_omega_to_v0;
      Matrix43r m_grad_quaternion_to_v0;
      Matrix3r m_grad_angle_in_radian_to_v0;
      Matrix3r m_grad_x_to_omega0;
      Matrix3r m_grad_v_to_omega0;
#endif 

		public:
			unsigned int numberOfParticles() const { return static_cast<unsigned int>(m_x.size()); }

			unsigned int getPointSetIndex() const { return m_pointSetIndex; }
			bool isSorted() const { return m_sorted; }

			void computeBoundaryVolume();
      void computeBoundaryPressureMLS();
      
			void resize(const unsigned int numBoundaryParticles);

			virtual void reset();

			virtual void performNeighborhoodSearchSort();

			virtual void saveState(BinaryFileWriter &binWriter);
			virtual void loadState(BinaryFileReader &binReader);

			void initModel(RigidBodyObject *rbo, const unsigned int numBoundaryParticles, Vector3r *boundaryParticles);
			
			FORCE_INLINE Vector3r &getPosition0(const unsigned int i)
			{
				return m_x0[i];
			}

			FORCE_INLINE const Vector3r &getPosition0(const unsigned int i) const
			{
				return m_x0[i];
			}

			FORCE_INLINE void setPosition0(const unsigned int i, const Vector3r &pos)
			{
				m_x0[i] = pos;
			}

			FORCE_INLINE Vector3r &getPosition(const unsigned int i)
			{
				return m_x[i];
			}

			FORCE_INLINE const Vector3r &getPosition(const unsigned int i) const
			{
				return m_x[i];
			}

			FORCE_INLINE void setPosition(const unsigned int i, const Vector3r &pos)
			{
				m_x[i] = pos;
			}

			FORCE_INLINE Vector3r &getVelocity(const unsigned int i)
			{
				return m_v[i];
			}

			FORCE_INLINE const Vector3r &getVelocity(const unsigned int i) const
			{
				return m_v[i];
			}

			FORCE_INLINE void setVelocity(const unsigned int i, const Vector3r &vel)
			{
				m_v[i] = vel;
			} 

			FORCE_INLINE const Real& getVolume(const unsigned int i) const
			{
				return m_V[i];
			}

			FORCE_INLINE Real& getVolume(const unsigned int i)
			{
				return m_V[i];
			}

			FORCE_INLINE void setVolume(const unsigned int i, const Real &val)
			{
				m_V[i] = val;
			}

			FORCE_INLINE Real& getPressure(const unsigned int i)
			{
				return m_pressure[i];
			}
			FORCE_INLINE const Real& getPressure(const unsigned int i) const
			{
				return m_pressure[i];
			}
      
      // -----------------------------------------------------------------------------------  
      FORCE_INLINE Vector3r get_position_rb()
      {
        return m_rigidBody->getPosition();
      }
      FORCE_INLINE Quaternionr get_quaternion_rb()
      {
        return m_rigidBody->getRotation();
      }
      FORCE_INLINE Vector4r get_quaternion_rb_vec4()
      {
        auto q = m_rigidBody->getRotation(); 
        return Vector4r(q.w(), q.x(), q.y(), q.z());
      }
      FORCE_INLINE Vector3r get_velocity_rb()
      {
        return m_rigidBody->getVelocity();
      }
      FORCE_INLINE Vector3r get_angular_velocity_rb()
      {
        return m_rigidBody->getAngularVelocity();
      }
      FORCE_INLINE void set_velocity_rb(const Vector3r new_v)
      {
        m_rigidBody->setVelocity(new_v);
      }
      FORCE_INLINE void set_angular_velocity_rb(const Vector3r new_omega)
      {
        m_rigidBody->setAngularVelocity(new_omega);
      }
      FORCE_INLINE void set_angular_velocity_rb_to_joint(const Vector3r new_omega)
      {
        m_rigidBody->setAngularVelocityToJoint(new_omega);
      }
      FORCE_INLINE void add_angular_velocity_rb_to_joint(const Vector3r delta_omega)
      {
        m_rigidBody->addAngularVelocityToJoint(delta_omega);
      }
      

#ifdef BACKWARD
			FORCE_INLINE Matrix3r& get_grad_force_to_omega(const unsigned int i)
			{
				return m_grad_force_to_omega[i];
			}
			FORCE_INLINE Matrix3r& get_grad_force_to_v(const unsigned int i)
			{
				return m_grad_force_to_v[i];
			}
			FORCE_INLINE Matrix3r& get_grad_force_to_x(const unsigned int i)
			{
				return m_grad_force_to_x[i];
			}
			FORCE_INLINE Matrix34r& get_grad_force_to_quaternion(const unsigned int i)
			{
				return m_grad_force_to_quaternion[i];
			}

      FORCE_INLINE Matrix3r& get_grad_torque_to_omega(const unsigned int i)
      {
        return m_grad_torque_to_omega[i];
      }
      FORCE_INLINE Matrix3r& get_grad_torque_to_v(const unsigned int i)
      {
          return m_grad_torque_to_v[i]; 
      }
      FORCE_INLINE Matrix3r& get_grad_torque_to_x(const unsigned int i)
      {
          return m_grad_torque_to_x[i]; 
      }
      FORCE_INLINE Matrix34r& get_grad_torque_to_quaternion(const unsigned int i)
      {
          return m_grad_torque_to_quaternion[i]; 
      }

		FORCE_INLINE const Matrix3r& get_grad_net_force_to_omega()
		{
			return grad_net_force_to_omega_n;
		}
		FORCE_INLINE const Matrix3r& get_grad_net_torque_to_omega()
		{
			return grad_net_torque_to_omega_n;
		}
		FORCE_INLINE const Matrix3r& get_grad_net_force_to_v()
		{
			return grad_net_force_to_vn;
		}
		FORCE_INLINE const Matrix3r& get_grad_net_torque_to_v()
		{
			return grad_net_torque_to_vn;
		}
	  // -----------------------------------------------------------------------------------  

			FORCE_INLINE Matrix3r get_grad_x_to_v0()
      {
          return m_grad_x_to_v0;
      }
			FORCE_INLINE Matrix3r get_grad_x_to_omega0()
      {
          return m_grad_x_to_omega0;
      }

			FORCE_INLINE Matrix43r get_grad_quaternion_to_omega0()
      {
          return m_grad_quaternion_to_omega0;
      }
			FORCE_INLINE Matrix43r get_grad_quaternion_to_v0()
      {
          return m_grad_quaternion_to_v0;
      }

			FORCE_INLINE Matrix3r get_grad_v_to_v0()
      {
          return m_grad_v_to_v0;
      }
			FORCE_INLINE Matrix3r get_grad_v_to_omega0()
      {
          return m_grad_v_to_omega0;
      }
			FORCE_INLINE Matrix3r get_grad_omega_to_omega0()
      {
          return m_grad_omega_to_omega0;
      }
			FORCE_INLINE Matrix3r get_grad_omega_to_v0()
      {
          return m_grad_omega_to_v0;
      }
      void reset_gradient(); // reset the gradient to the same as in 0-th timestep
      void accumulate_and_reset_gradient();
      void perform_chain_rule(const Real timeStep, const bool optimize_rotation);
      
      void update_rigid_body_gradient_manager(); 

      Matrix3r compute_grad_Rv_to_omega0(Vector3r v);
      Matrix3r compute_grad_RTv_to_omega0(Vector3r v);
      Matrix3r compute_grad_inertia_v_to_omega0(Vector3r v);

      Matrix3r compute_grad_Rv_to_v0(Vector3r v);
      Matrix3r compute_grad_RTv_to_v0(Vector3r v);
      Matrix3r compute_grad_inertia_v_to_v0(Vector3r v);
    //MatrixXr get_grad_rootRB_xn_to_actuator_omega0()
    //{
      //if(getRigidBodyObject() == getRigidBodyObject()->getSystem()->getRootRigidbody())
      //{
        ////return getRigidBodyObject()->getSystem()->get_grad_rootRB_xn_to_actuator_omega0();
      //}
      //else 
      //{
        //MatrixXr grad_rootRB_xn_to_actuator_omega0(this->getRigidBodyObject()->getSystem()->n_dof_actuator, 3);
        //grad_rootRB_xn_to_actuator_omega0.setZero();
        //return grad_rootRB_xn_to_actuator_omega0; 
      //}
   //}
		//Matrix3r get_grad_x_to_v0_another_bm(unsigned int index);
	  Matrix3r get_grad_x_to_omega0_another_bm(unsigned int index);
	  Matrix3r get_grad_x_to_v0_another_bm(unsigned int index);
    Matrix43r get_grad_quaternion_to_v0_another_bm(unsigned int index);
		//Matrix43r get_grad_quaternion_to_omega0_another_bm(unsigned int index);
		//Matrix3r get_grad_v_to_v0_another_bm(unsigned int index);
		//Matrix3r get_grad_v_to_omega0_another_bm(unsigned int index);
		//Matrix3r get_grad_omega_to_v0_another_bm(unsigned int index);
		//Matrix3r get_grad_omega_to_omega0_another_bm(unsigned int index);

		//Matrix3r get_grad_x_to_omega0_fixed_joint(unsigned int index);
		//Matrix43r get_grad_quaternion_to_omega0_fixed_joint(unsigned int index);
		//Matrix3r get_grad_v_to_omega0_fixed_joint(unsigned int index);
		//Matrix3r get_grad_omega_to_omega0_fixed_joint(unsigned int index);
#endif
	};
}

#endif
