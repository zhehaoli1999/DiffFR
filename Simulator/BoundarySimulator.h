#ifndef __BoundarySimulator_h__
#define __BoundarySimulator_h__

#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/InterlinkedSPH/RigidContactSolver.h"
//#include "SPlisHSPlasH/RigidBodyGradientManager.h"

namespace SPH
{
  class RigidContactSolver; 

	class BoundarySimulator 
	{
	public:
		BoundarySimulator() {}
		virtual ~BoundarySimulator() {}
		virtual void init() {}
		/** This function is called after the simulation scene is loaded and all
		* parameters are initialized. While reading a scene file several parameters
		* can change. The deferred init function should initialize all values which
		* depend on these parameters.
		*/
		virtual void deferredInit() {}
		virtual void timeStep() {}
    // separate the complete timeStep into velocity update + position integration
		virtual void velocityTimeStep() {} // for gradient computation 
		virtual void positionTimeStep() {} // for gradient computation
		virtual void updateVelocity() { updateBoundaryForces(); }
		virtual void initBoundaryData() {}
		virtual void reset() {}
    virtual void performNeighborhoodSearchSort() {}
    virtual RigidContactSolver* getRigidContactSolver() { return nullptr; }
    //virtual RigidBodyGradientManager* getRigidBodyGradientManager() { return nullptr; }

		void updateBoundaryForces();		
	};
}
 
#endif
