#ifndef __RigidBodyObject_h__
#define __RigidBodyObject_h__

#include "Common.h"
#include <vector>
#include "Simulator/ArticulatedSystemSimulator.h"

namespace SPH 
{	
	/** \brief Base class for rigid body objects. 
	*/
	class RigidBodyObject 
	{
	protected: 
		bool m_isAnimated; 

		ArticulatedDynamicSystemBase *system = nullptr;

	public:
		RigidBodyObject() { m_isAnimated = false;}
		virtual ~RigidBodyObject() {};

		void setArticulatedSystem(ArticulatedDynamicSystemBase *sys) { system = sys; }
		ArticulatedDynamicSystemBase *getArticulatedSystem() { return system; }
		
		virtual bool isDynamic() const = 0;
		bool isAnimated() const { return m_isAnimated; }
		virtual void setIsAnimated(const bool b) { m_isAnimated = b; }
		virtual void animate() {}

		virtual Real const getMass() const = 0;
		virtual Real const getInvMass() const {return 0.; }
    virtual Matrix3r const getInertiaTensorInverseWorld() const = 0; 
    virtual Matrix3r const getInertiaTensorWorld() const = 0; 
    virtual Matrix3r const getInertiaTensor0() const {return Matrix3r::Identity();}; 
		virtual Vector3r const& getPosition() const = 0;
    virtual Vector3r const& getPosition0() const = 0;
		virtual Vector3r const &getSystemPosition() const { return getPosition(); }
		virtual void setPosition(const Vector3r &x) = 0;
		virtual Vector3r getWorldSpacePosition() const = 0;
		virtual Vector3r const& getVelocity() const = 0;
		virtual Vector3r const &getVelocity0() const { return getVelocity(); }
		virtual void setVelocity(const Vector3r &v) = 0;
		virtual Quaternionr const& getRotation() const = 0;
		virtual Quaternionr const& getLastRotation() const = 0;
		virtual Quaternionr const& getRotation0() const = 0;
		virtual void setRotation(const Quaternionr &q) = 0;
		virtual Matrix3r getWorldSpaceRotation() const = 0;
		virtual Vector3r const& getAngularVelocity() const = 0;
		virtual Vector3r const &getAngularVelocity0() const { return getAngularVelocity(); }
		virtual void setAngularVelocity(const Vector3r &v) = 0;
		virtual void setAngularVelocityToJoint(const Vector3r &v) { setAngularVelocity(v); }
		virtual void addAngularVelocityToJoint(const Vector3r &delta_v) { setAngularVelocity(delta_v + getAngularVelocity()); }
		virtual void addForce(const Vector3r &f) = 0;
		virtual void addTorque(const Vector3r &t) = 0;
    virtual Vector3r getAngleDistanceInRadian() { return Vector3r::Zero(); }

		virtual void updateMeshTransformation() = 0;
		virtual const std::vector<Vector3r> &getVertices() const = 0;
		virtual const std::vector<Vector3r> &getVertexNormals() const = 0;
		virtual const std::vector<unsigned int> &getFaces() const = 0;
	};
}

#endif 
