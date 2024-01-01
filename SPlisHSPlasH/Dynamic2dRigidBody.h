#ifndef __DynamicRigidBody_h__
#define __DynamicRigidBody_h__

#include "Common.h"
#include "RigidBodyObject.h"
#include "TriangleMesh.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "Simulation.h"
#include "Simulator/Dynamic2dBoundarySimulator.h"

namespace SPH
{
	/** \brief This class stores the information of a 2d dynamic rigid body which
	* is not part of a rigid body simulation.
	*/
	class Dynamic2dRigidBody : public RigidBodyObject
	{
	protected:
		Vector3r m_x0;
		Vector3r m_x;
		Quaternionr m_q;
		Quaternionr m_last_q;
		Quaternionr m_q0;
		Vector3r m_velocity0;
		Vector3r m_angularVelocity0;
		Vector3r m_velocity;
		Vector3r m_angularVelocity;
		TriangleMesh m_geometry;

		Real m_mass;
		Real m_invMass;
		Real m_inertia;
		Real m_invInertia;

		bool m_isDynamic;

    Vector3r m_angleDistanceInRadian;

	public:
		Dynamic2dRigidBody()
		{
			m_isAnimated = false;
			m_velocity = Vector3r::Zero();
			m_angularVelocity = Vector3r::Zero();
			m_velocity0 = Vector3r::Zero();
			m_angularVelocity0 = Vector3r::Zero();
      m_angleDistanceInRadian = Vector3r::Zero();

			m_mass = 0;
			m_invMass = 0;
			m_inertia = 0;
			m_invInertia = 0;
		}

		virtual bool isDynamic() const { return m_isDynamic; }
		virtual void setIsDynamic(bool b) { m_isDynamic = b; }

		virtual Real const getMass() const { return m_mass; }
    virtual Matrix3r const getInertiaTensorInverseWorld() const {return Matrix3r::Identity() * m_invInertia;}
    virtual Matrix3r const getInertiaTensorWorld() const {return Matrix3r::Identity() * m_inertia;}
    virtual Matrix3r const getInertiaTensor0() const {return  m_inertia * Matrix3r::Identity() ;}
		virtual Real const getInvMass() const{ return m_invMass; }
		virtual Vector3r const& getPosition() const { return m_x; }
		virtual Vector3r const &getSystemPosition() const
		{
			if (!dynamic_cast<articulatedDynamic2dSystem *>(system))
				return getPosition();
			else
				return dynamic_cast<articulatedDynamic2dSystem *>(system)->getPosition();
		}
		virtual void setPosition(const Vector3r& x) { m_x = x; updateMeshTransformation(); }
		Vector3r const& getPosition0() const { return m_x0; }
		void setPosition0(const Vector3r& x) { m_x0 = x; }
		virtual Vector3r getWorldSpacePosition() const { return m_x; }
		virtual Vector3r const& getVelocity() const { return m_velocity; }
		virtual void setVelocity(const Vector3r& v) { m_velocity = v; m_velocity[2] = 0.; }
		virtual Vector3r const& getVelocity0() const { return m_velocity0; }
		virtual void setVelocity0(const Vector3r& v) { m_velocity0 = v; m_velocity0[2] = 0.; }
		virtual Quaternionr const& getRotation() const { return m_q; }
		virtual Quaternionr const& getLastRotation() const { return m_last_q; }
		virtual void setRotation(const Quaternionr& q) { m_q = q; m_last_q = m_q; updateMeshTransformation(); }
		Quaternionr const& getRotation0() const { return m_q0; }
		void setRotation0(const Quaternionr& q) { m_q0 = q; }
		virtual Matrix3r getWorldSpaceRotation() const { return m_q.toRotationMatrix(); }
    virtual Vector3r getAngleDistanceInRadian() { return m_angleDistanceInRadian; }
		virtual Vector3r const& getAngularVelocity() const { return m_angularVelocity; }
		virtual void setAngularVelocity(const Vector3r& v) { m_angularVelocity = v; m_angularVelocity[0] = 0; m_angularVelocity[1] = 0;}
		virtual Vector3r const& getAngularVelocity0() const { return m_angularVelocity0; }
		virtual void setAngularVelocity0(const Vector3r& v) { m_angularVelocity0 = v; m_angularVelocity0[0] = 0; m_angularVelocity0[1] = 0;}
		virtual void setAngularVelocityToJoint(const Vector3r &v)
		{
			if(dynamic_cast<articulatedDynamic2dSystem*>(system))
			{
				dynamic_cast<articulatedDynamic2dSystem*>(system)->setAngularVelocityToJoint(this, v);
			}
			else
			{
				std::cout << "Warning: No joints for rigid body";
			}
		}
		virtual void addAngularVelocityToJoint(const Vector3r &v)
		{
			if(dynamic_cast<articulatedDynamic2dSystem*>(system))
			{
				dynamic_cast<articulatedDynamic2dSystem*>(system)->addAngularVelocityToJoint(this, v);
			}
			else
			{
				std::cout << "Warning: No joints for rigid body";
			}
		}
		virtual void addForce(const Vector3r& f)
		{
			if (dynamic_cast<articulatedDynamic2dSystem*>(system))
			{
				dynamic_cast<articulatedDynamic2dSystem*>(system)->addForce(f, this);
			}
			else
			{
				const Real dt = SPH::TimeManager::getCurrent()->getTimeStepSize();
				m_velocity += m_invMass * f * dt;
				m_velocity[2] = 0; // set z axis velocity to be 0 in 2d case
			}
		}
		virtual void addTorque(const Vector3r& t)
		{
			if(dynamic_cast<articulatedDynamic2dSystem*>(system))
			{
				dynamic_cast<articulatedDynamic2dSystem*>(system)->addTorque(t, this);
			}
			else
			{
				const Real dt = SPH::TimeManager::getCurrent()->getTimeStepSize();
				m_angularVelocity += m_invInertia * t * dt;
				m_angularVelocity[0] = m_angularVelocity[1] = 0;
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
      m_last_q =  m_q;
			m_velocity = m_velocity0;
			m_angularVelocity = m_angularVelocity0;
      m_angleDistanceInRadian.setZero();
			updateMeshTransformation();
		}

		void determineMassProperties(Real density, Real radius, const std::vector<Vector3r>& boundaryParticles)
		{
      //TODO: modify this to directly set mass in gui 
			//if (density < 1000) density = 1000;
			Real volume = radius * radius;
			Real deltaMass = volume * density;
			// here calculations can be improved
			for (const auto& particle : boundaryParticles)
			{
				m_mass += deltaMass;
				m_inertia += deltaMass * (particle).squaredNorm();
			}
			m_invMass = 1. / m_mass;
			m_invInertia = 1. / m_inertia;
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
