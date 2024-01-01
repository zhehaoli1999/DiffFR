#ifndef __2DDynamicBoundarySimulator_h__
#define __2DDynamicBoundarySimulator_h__

#include "Simulator/BoundarySimulator.h"
#include "Simulator/SimulatorBase.h"
#include "Simulator/ArticulatedSystemSimulator.h"
#include "SPlisHSPlasH/InterlinkedSPH/RigidContactSolver.h"
#include <memory>

namespace SPH
{
	/* This class is created as a temporary solution to perform 2d dynamic boundary simulation.
	* Since PBD cannot be directly used, here we ignore all the rigid-rigid collisions for simplication.
	*/
	class Dynamic2dRigidBody;

	class articulatedDynamic2dSystem : public ArticulatedDynamicSystemBase
	{
	};

	class Dynamic2dBoundarySimulator : public BoundarySimulator
	{
	protected:
		SimulatorBase* m_base;

		std::vector<articulatedDynamic2dSystem *> articulateSystems;

		void loadObj(const std::string& filename, TriangleMesh& mesh, const Vector3r& scale);

    std::unique_ptr<RigidContactSolver> m_rigid_contact_solver; 

	public:
		Dynamic2dBoundarySimulator(SimulatorBase* base);
		virtual ~Dynamic2dBoundarySimulator();

		virtual void init() override;
		/** This function is called after the simulation scene is loaded and all
		* parameters are initialized. While reading a scene file several parameters
		* can change. The deferred init function should initialize all values which
		* depend on these parameters.
		*/
		virtual void deferredInit() override;
		virtual void timeStep() override;
		virtual void velocityTimeStep() override;
		virtual void positionTimeStep() override;
		virtual void initBoundaryData() override;
		virtual void reset() override;
    virtual RigidContactSolver* getRigidContactSolver() override;
    virtual void performNeighborhoodSearchSort() override; 
	};
}

#endif
