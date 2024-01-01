#ifndef __DynamicRigidBody_h__
#define __DynamicRigidBody_h__

#include "Common.h"
#include "RigidBodyObject.h"
#include "TriangleMesh.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "Simulation.h"
#include "Simulator/RigidBody3dBoundarySimulator.h"

namespace SPH
{
	/** \brief This class stores the information of a 3d dynamic rigid body which
	* is not part of a rigid body simulation.
	*/
	class Dynamic3dRigidBody : public RigidBodyObject
	{
	protected:
		Vector3r m_x0;
		Vector3r m_x;
		Quaternionr m_q;
		Quaternionr m_last_q;
		Quaternionr m_q0;
		Vector3r m_velocity;
		Vector3r m_velocity0;
		Vector3r m_angularVelocity;
    Vector3r m_angularVelocity0;
		TriangleMesh m_geometry;

		Real m_mass;
		Real m_invMass;
		Matrix3r m_inertia0;
		Matrix3r m_inertia;
		Matrix3r m_invInertia;

		bool m_isDynamic;

    Vector3r m_angleDistanceInRadian;

	public:
		Dynamic3dRigidBody()
		{
			m_isAnimated = false;
			m_velocity = Vector3r::Zero();
			m_velocity0 = Vector3r::Zero();
			m_angularVelocity = Vector3r::Zero();
			m_angularVelocity0 = Vector3r::Zero();

			m_mass = 0;
			m_invMass = 0;
			m_inertia0 = Matrix3r::Zero();
			m_inertia = Matrix3r::Zero();
			m_invInertia = Matrix3r::Zero();
		}

		virtual bool isDynamic() const { return m_isDynamic; }
		virtual void setIsDynamic(bool b) { m_isDynamic = b; }

		virtual Real const getMass() const { return m_mass; }
    virtual Matrix3r const getInertiaTensorInverseWorld() const {return  m_invInertia;}
    virtual Matrix3r const getInertiaTensorWorld() const {return  m_inertia;}
    virtual Matrix3r const getInertiaTensor0() const {return  m_inertia0;}
		virtual Real const getInvMass() const{ return m_invMass; }
		virtual Vector3r const& getPosition() const { return m_x; }
		virtual Vector3r const &getSystemPosition() const
		{
			if (!dynamic_cast<articulatedDynamic3dSystem *>(system))
				return getPosition();
			else
				return dynamic_cast<articulatedDynamic3dSystem *>(system)->getPosition();
		}
		virtual void setPosition(const Vector3r& x) { m_x = x; updateMeshTransformation(); }
		Vector3r const& getPosition0() const { return m_x0; }
		void setPosition0(const Vector3r& x) { m_x0 = x; }
		virtual Vector3r getWorldSpacePosition() const { return m_x; }
		virtual Vector3r const& getVelocity() const { return m_velocity; }
		virtual Vector3r const& getVelocity0() const { return m_velocity0; };
		virtual void setVelocity(const Vector3r& v) { m_velocity = v; }
		virtual void setVelocity0(const Vector3r& v) { m_velocity0 = v; }
		virtual Quaternionr const& getRotation() const { return m_q; }
		virtual Quaternionr const& getLastRotation() const { return m_last_q; }
		virtual void setRotation(const Quaternionr& q) { m_q = q; m_last_q = m_q; updateMeshTransformation(); }
		Quaternionr const& getRotation0() const { return m_q0; }
		void setRotation0(const Quaternionr& q) { m_q0 = q; }
		virtual Matrix3r getWorldSpaceRotation() const { return m_q.toRotationMatrix(); }
    virtual Vector3r getAngleDistanceInRadian() { return m_angleDistanceInRadian; }
		virtual Vector3r const& getAngularVelocity() const { return m_angularVelocity; }
		virtual Vector3r const& getAngularVelocity0() const { return m_angularVelocity0; }
		virtual void setAngularVelocity(const Vector3r& v) { m_angularVelocity = v; }
    virtual void setAngularVelocity0(const Vector3r& v) { m_angularVelocity0 = v;} 
		virtual void setAngularVelocityToJoint(const Vector3r &v)
		{
			if(dynamic_cast<articulatedDynamic3dSystem*>(system))
			{
				dynamic_cast<articulatedDynamic3dSystem*>(system)->setAngularVelocityToJoint(this, v);
			}
			else
			{
				std::cout << "Warning: No joints for rigid body";
			}
		}
		virtual void addAngularVelocityToJoint(const Vector3r &v)
		{
			if(dynamic_cast<articulatedDynamic3dSystem*>(system))
			{
				dynamic_cast<articulatedDynamic3dSystem*>(system)->addAngularVelocityToJoint(this, v);
			}
			else
			{
				std::cout << "Warning: No joints for rigid body";
			}
		}
		virtual void addForce(const Vector3r& f)
		{ 
			if (dynamic_cast<articulatedDynamic3dSystem*>(system))
			{
				dynamic_cast<articulatedDynamic3dSystem*>(system)->addForce(f, this);
			}
			else
			{
			  const Real dt = SPH::TimeManager::getCurrent()->getTimeStepSize();
			  m_velocity += m_invMass * f * dt;
		  }
    }
		virtual void addTorque(const Vector3r& t)
		{
			if(dynamic_cast<articulatedDynamic3dSystem*>(system))
			{
				dynamic_cast<articulatedDynamic3dSystem*>(system)->addTorque(t, this);
			}
			else
			{
			  const Real dt = SPH::TimeManager::getCurrent()->getTimeStepSize();
        if(Simulation::getCurrent()->getRigidBodyMode() == static_cast<int>(RigidBodyMode::WithGyroscopic))
        {
          Vector3r L = m_inertia * m_angularVelocity; 
          m_angularVelocity += m_invInertia * ( L.cross(m_angularVelocity) +  t )* dt;  
        }
        else
          m_angularVelocity += m_invInertia * t * dt; 
		  }
    }
		void animate()
		{
			const Real dt = TimeManager::getCurrent()->getTimeStepSize();
			m_x += m_velocity * dt;
			Quaternionr angVelQ(0.0, m_angularVelocity[0], m_angularVelocity[1], m_angularVelocity[2]);
      m_last_q = m_q;
			m_q.coeffs() += dt * 0.5 * (angVelQ * m_q).coeffs();
			m_q.normalize();
      updateMeshTransformation();
      updateInertia();
      m_angleDistanceInRadian += dt * m_angularVelocity;
		}

		virtual const std::vector<Vector3r>& getVertices() const { return m_geometry.getVertices(); };
		virtual const std::vector<Vector3r>& getVertexNormals() const { return m_geometry.getVertexNormals(); };
		virtual const std::vector<unsigned int>& getFaces() const { return m_geometry.getFaces(); };

		void setWorldSpacePosition(const Vector3r& x) { setPosition(x); }
		void setWorldSpaceRotation(const Matrix3r& r) { setRotation(Quaternionr(r)); }
		TriangleMesh& getGeometry() { return m_geometry; }

		virtual void updateMeshTransformation()
		{
			m_geometry.updateMeshTransformation(m_x, m_q.toRotationMatrix());
			m_geometry.updateNormals();
			m_geometry.updateVertexNormals();
		}

		void reset()
		{
			m_x = m_x0;
			m_q = m_q0;
      m_last_q = m_q;
      m_inertia = m_inertia0;
      updateInertia(); 
      m_velocity = m_velocity0;
      m_angularVelocity = m_angularVelocity0;
      m_angleDistanceInRadian.setZero();
			updateMeshTransformation();
		}

		void determineMassProperties(Real density, Real radius, const std::vector<Vector3r>& boundaryParticles)
		{
			Real volume =static_cast<Real>(4.0 / 3.0 * M_PI) * radius * radius * radius; //TODO
			Real deltaMass = volume * density;
      m_mass = 0.; 
      m_inertia0.setZero(); 
			// here calculations can be improved
			for (const auto& x_particle : boundaryParticles)
			{
				m_mass += deltaMass;
        Vector3r r = x_particle;
				m_inertia0 += deltaMass * (r.dot(r) * Matrix3r::Identity() - r * r.transpose());
			}
			m_invMass = 1. / m_mass;
      updateInertia();
      if(isDynamic())
      {
        
      std::cout << "mass rb = " << m_mass << std::endl;
      std::cout << "m_inertia0 = " << m_inertia0 << std::endl;
      std::cout << "m_inertia = " << m_inertia << std::endl;
      //Matrix3d m_inertia_double = m_inertia.cast<double>();
      std::cout << "m_invInertia = " << m_invInertia << std::endl;
      //std::cout << "m_invInertia_double = " << m_inertia_double.inverse() << std::endl;
      //std::cout << "m_invInertia2= " << static_cast<Eigen::Matrix3d>(m_inertia).inverse() << std::endl;
      //std::cout << "m_invInertia2 @ m_inertia = " << m_inertia * static_cast<Eigen::Matrix3d>(m_inertia).inverse()<< std::endl;
      //std::cout << "m_inertia @ m_inertia = " << m_inertia * m_inertia<< std::endl;
      std::cout << "m_invInertia @ m_inertia = " << m_invInertia * m_inertia << std::endl;
      }
		}

    void updateInertia()
    {
      m_inertia = m_q.toRotationMatrix() * m_inertia0 * m_q.toRotationMatrix().transpose();
      m_invInertia = m_inertia.inverse();
    }

		void moveWithGravity(Real dt)
		{  
			if (m_isDynamic)
			{
        addGravity();
			  animate();
      }
		}

    void addGravity()
    {
			Simulation *sim = Simulation::getCurrent();
			const Vector3r gravity(sim->getVecValue<Real>(Simulation::GRAVITATION));
			this->addForce(gravity * m_mass);
		}
	};
}

#endif 
