#pragma once 

#include "SPlisHSPlasH/Common.h"
#include "Simulator/ArticulatedSystemSimulator.h"
#include "Simulator/BoundarySimulator.h"
#include "Simulator/SimulatorBase.h"
#include "SPlisHSPlasH/InterlinkedSPH/RigidContactSolver.h"

namespace SPH
{
  class Dynamic3dRigidBody;  

  class articulatedDynamic3dSystem: public ArticulatedDynamicSystemBase
  {
  };

   /* This class is used to simulate 3d rigidbody motion to replace PBD lib  
    *
    */
	class RigidBody3dBoundarySimulator : public BoundarySimulator
	{
	protected:
		SimulatorBase* m_base;

		std::vector<articulatedDynamic3dSystem *> articulateSystems;

		void loadObj(const std::string& filename, TriangleMesh& mesh, const Vector3r& scale);

    std::unique_ptr<RigidContactSolver> m_rigid_contact_solver; 

	public:
		RigidBody3dBoundarySimulator(SimulatorBase* base);
		virtual ~RigidBody3dBoundarySimulator();

		virtual void init();
		/** This function is called after the simulation scene is loaded and all
		* parameters are initialized. While reading a scene file several parameters
		* can change. The deferred init function should initialize all values which
		* depend on these parameters.
		*/
		virtual void deferredInit();
		virtual void timeStep() override;
		virtual void velocityTimeStep() override;
		virtual void positionTimeStep() override;
		virtual void initBoundaryData();
		virtual void reset();
    virtual RigidContactSolver* getRigidContactSolver();
    virtual void performNeighborhoodSearchSort(); 
	};
}
