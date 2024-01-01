#include "Dynamic2dBoundarySimulator.h"
#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/InterlinkedSPH/RigidContactSolver.h"
#include "SimulatorBase.h"
#include "Utilities/ColorfulPrint.h"
#include "Utilities/FileSystem.h"
#include "SPlisHSPlasH/Simulation.h"
#include "Utilities/PartioReaderWriter.h"
#include "SPlisHSPlasH/Dynamic2dRigidBody.h"
#include "Utilities/Logger.h"
#include "Utilities/Timing.h"
#include "SPlisHSPlasH/Utilities/SurfaceSampling.h"
#include "Utilities/OBJLoader.h"
#include "SPlisHSPlasH/TriangleMesh.h"
#include "Simulator/SceneConfiguration.h"

using namespace std;
using namespace SPH;
using namespace Utilities;

Dynamic2dBoundarySimulator::Dynamic2dBoundarySimulator(SimulatorBase *base)
{
	m_base = base;
  m_rigid_contact_solver = nullptr; 
}

Dynamic2dBoundarySimulator::~Dynamic2dBoundarySimulator()
{
	for(auto&& system : this->articulateSystems)
		delete system;
}

void Dynamic2dBoundarySimulator::loadObj(const std::string& filename, TriangleMesh& mesh, const Vector3r& scale)
{
	std::vector<OBJLoader::Vec3f> x;
	std::vector<OBJLoader::Vec3f> normals;
	std::vector<MeshFaceIndices> faces;
	OBJLoader::Vec3f s = { (float)scale[0], (float)scale[1], (float)scale[2] };
	OBJLoader::loadObj(filename, &x, &faces, &normals, nullptr, s);

	mesh.release();
	const unsigned int nPoints = (unsigned int)x.size();
	const unsigned int nFaces = (unsigned int)faces.size();
	mesh.initMesh(nPoints, nFaces);
	for (unsigned int i = 0; i < nPoints; i++)
	{
		mesh.addVertex(Vector3r(x[i][0], x[i][1], x[i][2]));
	}
	for (unsigned int i = 0; i < nFaces; i++)
	{
		// Reduce the indices by one
		int posIndices[3];
		for (int j = 0; j < 3; j++)
		{
			posIndices[j] = faces[i].posIndices[j] - 1;
		}

		mesh.addFace(&posIndices[0]);
	}

	LOG_INFO << "Number of triangles: " << nFaces;
	LOG_INFO << "Number of vertices: " << nPoints;
}

void Dynamic2dBoundarySimulator::init()
{
}

void Dynamic2dBoundarySimulator::initBoundaryData()
{
	// don't allow any cache
	const std::string& sceneFile = SceneConfiguration::getCurrent()->getSceneFile();
	const Utilities::SceneLoader::Scene& scene = SceneConfiguration::getCurrent()->getScene();
	std::string scene_path = FileSystem::getFilePath(sceneFile);
	std::string scene_file_name = FileSystem::getFileName(sceneFile);
	Simulation* sim = Simulation::getCurrent();

	for (unsigned int i = 0; i < scene.boundaryModels.size(); i++)
	{
		string meshFileName = scene.boundaryModels[i]->meshFile;
		if (FileSystem::isRelativePath(meshFileName))
			meshFileName = FileSystem::normalizePath(scene_path + "/" + scene.boundaryModels[i]->meshFile);

		Dynamic2dRigidBody* rb = new Dynamic2dRigidBody();
		rb->setIsAnimated(scene.boundaryModels[i]->isAnimated);
		rb->setIsDynamic(scene.boundaryModels[i]->dynamic);
		TriangleMesh& geo = rb->getGeometry();
		loadObj(meshFileName, geo, scene.boundaryModels[i]->scale);

		std::vector<Vector3r> boundaryParticles;
		if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
		{
			LOG_INFO << "2D regular sampling of " << meshFileName;
			START_TIMING("2D regular sampling");
			RegularSampling2D sampling;
			sampling.sampleMesh(Matrix3r::Identity(), Vector3r::Zero(),
				geo.numVertices(), geo.getVertices().data(), geo.numFaces(),
				geo.getFaces().data(), 1.75f * scene.particleRadius, boundaryParticles);
			STOP_TIMING_AVG;
		}

		Quaternionr q(scene.boundaryModels[i]->rotation);
		rb->setPosition0(scene.boundaryModels[i]->translation);
		rb->setPosition(scene.boundaryModels[i]->translation);
		rb->setRotation0(q);
		rb->setRotation(q);
		//rb->setVelocity0(scene.boundaryModels[i]->init_velocity);
		//rb->setVelocity(scene.boundaryModels[i]->init_velocity);
		//rb->setAngularVelocity0(scene.boundaryModels[i]->init_angular_velocity);
		//rb->setAngularVelocity(scene.boundaryModels[i]->init_angular_velocity);
		rb->determineMassProperties(scene.boundaryModels[i]->density, scene.particleRadius, boundaryParticles);

		if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
		{
			BoundaryModel_Akinci2012* bm = new BoundaryModel_Akinci2012();
			bm->initModel(rb, static_cast<unsigned int>(boundaryParticles.size()), &boundaryParticles[0]);
			sim->addBoundaryModel(bm);
		}
		else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
		{
			BoundaryModel_Koschier2017* bm = new BoundaryModel_Koschier2017();
			bm->initModel(rb);
			sim->addBoundaryModel(bm);
			SPH::TriangleMesh& mesh = rb->getGeometry();
			m_base->initDensityMap(mesh.getVertices(), mesh.getFaces(), scene.boundaryModels[i], false, true, bm);
		}
		else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
		{
			BoundaryModel_Bender2019* bm = new BoundaryModel_Bender2019();
			bm->initModel(rb);
			sim->addBoundaryModel(bm);
			SPH::TriangleMesh& mesh = rb->getGeometry();
			m_base->initVolumeMap(mesh.getVertices(), mesh.getFaces(), scene.boundaryModels[i], false, true, bm);
		}
		rb->updateMeshTransformation();
	}

	for (unsigned int i = 0; i < scene.articulatedSystems.size();i++)
	{
    // TODO: replace new with smart pointers 
		auto artSystem = new articulatedDynamic2dSystem();
		auto &artSystemData = scene.articulatedSystems[i];
		for(auto index : artSystemData->rigidBodyIndices)
		{
			auto bm = dynamic_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModel(index));
			assert(bm && "Articulated systems are now allowed only for 2012 Akinci boundary handling.\n");
			auto rbo = dynamic_cast<Dynamic2dRigidBody *>(bm->getRigidBodyObject());
			assert(rbo && "2d articulated systems are only allowed for 2d dynamic simulations.");
			artSystem->addRigidBody(rbo);
			rbo->setArticulatedSystem(artSystem);
			artSystem->setLastRigidBoundaryModel(bm);
		}
		for(auto&joint : artSystemData->joints)
		{
			artSystem->addJoint(joint.rigidIndex1, joint.rigidIndex2, joint.jointPoint, joint.isActuator);
		}
		artSystem->inv_stiffness = artSystemData->inv_stiffness;
		artSystem->init();
		articulateSystems.push_back(artSystem);
	}

}

void Dynamic2dBoundarySimulator::deferredInit()
{
	Simulation* sim = Simulation::getCurrent();
  if(sim->useRigidContactSolver()) 
  {
    // This should be before performNeighborhoodSearchSort
    m_rigid_contact_solver = std::unique_ptr<RigidContactSolver>(new RigidContactSolver());   
  }

	sim->performNeighborhoodSearchSort();
	if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
	{
		m_base->updateBoundaryParticles(true);
		Simulation::getCurrent()->updateBoundaryVolume(); // Seems the actual neighborhood search is happened here
	}
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
		m_base->updateDMVelocity();
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
		m_base->updateVMVelocity();


}

void Dynamic2dBoundarySimulator::timeStep()
{
  velocityTimeStep();
  positionTimeStep();
}

void Dynamic2dBoundarySimulator::velocityTimeStep()
{
	Simulation *sim = Simulation::getCurrent();
  Real h = TimeManager::getCurrent()->getTimeStepSize();
  const unsigned int nObjects = sim->numberOfBoundaryModels();

  if(sim->useRigidContactSolver()) // for interlinked SPH solver 
  {
    for (unsigned int i = 0; i < nObjects; i++)
    {
      BoundaryModel *bm = sim->getBoundaryModel(i);
      RigidBodyObject *rbo = bm->getRigidBodyObject();
      auto d2drbo = dynamic_cast<Dynamic2dRigidBody *>(rbo);
      if (d2drbo && rbo->isDynamic())
      {
        d2drbo->addGravity(); // FIXME: move this gravity velocity update before fluid solver 
      }
    }

    m_rigid_contact_solver->solveRigidContact(); 
  }
  else { // Do not use rigid contact solver 
    updateBoundaryForces();

    for (unsigned int i = 0; i < nObjects; i++)
    {
      BoundaryModel *bm = sim->getBoundaryModel(i);
      RigidBodyObject *rbo = bm->getRigidBodyObject();
      auto d2drbo = dynamic_cast<Dynamic2dRigidBody *>(rbo);
      if (d2drbo && rbo->isDynamic())
      {
        d2drbo->addGravity(); // FIXME: this should before the fluid timestep? 
      }
    }
    for (auto &system : articulateSystems)
    {
      //LOG_INFO << CyanHead() <<  "size of articulatedSystems = " << articulateSystems.size() << CyanTail(); 
      system->timeStep();
      //system->checkJointsStatus();
    }

   } // end else
}

void Dynamic2dBoundarySimulator::positionTimeStep()
{
	Simulation *sim = Simulation::getCurrent();
  Real h = TimeManager::getCurrent()->getTimeStepSize();
  const unsigned int nObjects = sim->numberOfBoundaryModels();

  for (unsigned int i = 0; i < nObjects; i++)
  {
    BoundaryModel *bm = sim->getBoundaryModel(i);
    RigidBodyObject *rbo = bm->getRigidBodyObject();
    auto d2drbo = dynamic_cast<Dynamic2dRigidBody *>(rbo);
    if (d2drbo && rbo->isDynamic())
    {
      d2drbo->animate();
    }
  }

  if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
    m_base->updateBoundaryParticles(false);
  else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
    m_base->updateDMVelocity();
  else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
    m_base->updateVMVelocity();
}

void Dynamic2dBoundarySimulator::reset()
{
	Simulation* sim = Simulation::getCurrent();
	for (unsigned int i = 0; i < sim->numberOfBoundaryModels(); i++)
	{
		BoundaryModel* bm = sim->getBoundaryModel(i);
		((Dynamic2dRigidBody*)bm->getRigidBodyObject())->reset();
	}
	for (auto &system : articulateSystems)
	{
		system->reset();
	}

  LOG_INFO << "!!!! before contact solver reset";
  if(m_rigid_contact_solver != nullptr)
    m_rigid_contact_solver->reset(); 
  LOG_INFO << "!!!! contact solver reset";

	//sim->performNeighborhoodSearchSort();
	if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
	{
		m_base->updateBoundaryParticles(true);
    Simulation::getCurrent()->updateBoundaryVolume(); // ZhehaoLi: Note: need to add this to prevent penetration
	}
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
		m_base->updateDMVelocity();
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
		m_base->updateVMVelocity();
}

RigidContactSolver* Dynamic2dBoundarySimulator::getRigidContactSolver()
{
  return m_rigid_contact_solver.get(); 
}

void Dynamic2dBoundarySimulator::performNeighborhoodSearchSort()
{
  if(Simulation::getCurrent()->useRigidContactSolver())
    m_rigid_contact_solver->performNeighborhoodSearchSort(); 
}
