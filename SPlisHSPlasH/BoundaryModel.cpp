#include "BoundaryModel.h"
#include "SPHKernels.h"
#include <iostream>
#include "TimeManager.h"
#include "TimeStep.h"
#include "Utilities/Logger.h"
#include "NeighborhoodSearch.h"
#include "Simulation.h"

using namespace SPH;


BoundaryModel::BoundaryModel() :
  m_forcePerThread(),
  m_torquePerThread(),
  m_forcePerThread_backup(),
  m_torquePerThread_backup()
{
}

BoundaryModel::~BoundaryModel(void)
{
  m_forcePerThread.clear();
  m_torquePerThread.clear();

  delete m_rigidBody;
}

void BoundaryModel::reset()
{
  for (int j = 0; j < m_forcePerThread.size(); j++)
    {
      m_forcePerThread[j].setZero();
      m_torquePerThread[j].setZero();
    }
}

void BoundaryModel::getForceAndTorque(Vector3r &force, Vector3r &torque)
{
  force.setZero();
  torque.setZero();
  for (int j = 0; j < m_forcePerThread.size(); j++)
  {
    //TODO: add backward here
    force += m_forcePerThread[j];
    torque += m_torquePerThread[j];

    m_forcePerThread_backup[j] = m_forcePerThread[j];
    m_torquePerThread_backup[j] = m_torquePerThread[j];
  }
  //auto sim = Simulation::getCurrent();
  //if(sim->isDebug())
  //std::cout << "!!!!!!!!!!! [rbo addForce] f = " << force.transpose()<< std::endl;
}
Vector3r BoundaryModel::getForce()
{
  Vector3r force = Vector3r::Zero();
  for (int j = 0; j < m_forcePerThread_backup.size(); j++)
  {
      force += m_forcePerThread_backup[j];
  }
  return force; 
}
Vector3r BoundaryModel::getTorque()
{
  Vector3r torque = Vector3r::Zero();
  for (int j = 0; j < m_forcePerThread_backup.size(); j++)
  {
      torque += m_torquePerThread_backup[j];
  }
  return torque; 
}


void BoundaryModel::clearForceAndTorque()
{
  for (int j = 0; j < m_forcePerThread.size(); j++)
    {
      m_forcePerThread_backup[j] = m_forcePerThread[j];
      m_torquePerThread_backup[j] = m_torquePerThread[j];

      m_forcePerThread[j].setZero();
      m_torquePerThread[j].setZero();
    }
}
