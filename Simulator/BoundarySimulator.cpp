#include "BoundarySimulator.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "SPlisHSPlasH/Simulation.h"
#include "SPlisHSPlasH/BoundaryModel.h"
#include "Utilities/Logger.h"
#include "Utilities/ColorfulPrint.h"

using namespace SPH;

void BoundarySimulator::updateBoundaryForces()
{
  Real h = TimeManager::getCurrent()->getTimeStepSize();
  Simulation *sim = Simulation::getCurrent();
  const unsigned int nObjects = sim->numberOfBoundaryModels();
  for (unsigned int i = 0; i < nObjects; i++)
    {
      BoundaryModel *bm = sim->getBoundaryModel(i);
      RigidBodyObject *rbo = bm->getRigidBodyObject();
      if (rbo->isDynamic())
      {
        if(false == rbo->isAnimated())
        {
          Vector3r force, torque;
          bm->getForceAndTorque(force, torque);
          //if(sim->isDebug()){
            //LOG_INFO << Utilities::YellowHead() <<  "net force = " << force.transpose() << Utilities::YellowTail() << "\n";
            //LOG_INFO << Utilities::YellowHead() <<  "net torque = " << torque.transpose() << Utilities::YellowTail() << "\n";
          //}
          rbo->addForce(force);
          rbo->addTorque(torque);
        }

        bm->clearForceAndTorque();
      }
    }
}
