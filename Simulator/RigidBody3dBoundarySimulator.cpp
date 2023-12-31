#include "RigidBody3dBoundarySimulator.h"
#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/InterlinkedSPH/RigidContactSolver.h"
#include "SimulatorBase.h"
#include "Utilities/FileSystem.h"
#include "SPlisHSPlasH/Simulation.h"
#include "Utilities/PartioReaderWriter.h"
#include "SPlisHSPlasH/Dynamic3dRigidBody.h"
#include "Utilities/Logger.h"
#include "Utilities/Timing.h"
#include "SPlisHSPlasH/Utilities/SurfaceSampling.h"
#include "Utilities/OBJLoader.h"
#include "SPlisHSPlasH/TriangleMesh.h"
#include "Simulator/SceneConfiguration.h"

using namespace std;
using namespace SPH;
using namespace Utilities;

RigidBody3dBoundarySimulator::RigidBody3dBoundarySimulator(SimulatorBase* base)
{
  m_base = base; 
  m_rigid_contact_solver = nullptr; 
}

RigidBody3dBoundarySimulator::~RigidBody3dBoundarySimulator()
{

}


void RigidBody3dBoundarySimulator::loadObj(const std::string& filename, TriangleMesh& mesh, const Vector3r& scale)
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

void RigidBody3dBoundarySimulator::init()
{
}

void RigidBody3dBoundarySimulator::initBoundaryData()
{
	const std::string& sceneFile = SceneConfiguration::getCurrent()->getSceneFile();
	const Utilities::SceneLoader::Scene& scene = SceneConfiguration::getCurrent()->getScene();
	std::string scene_path = FileSystem::getFilePath(sceneFile);
	std::string scene_file_name = FileSystem::getFileName(sceneFile);
	// no cache for 2D scenes
	// 2D sampling is fast, but storing it would require storing the transformation as well
	const bool useCache = m_base->getUseParticleCaching() && !scene.sim2D;
	Simulation *sim = Simulation::getCurrent();

	string cachePath = scene_path + "/Cache";

	for (unsigned int i = 0; i < scene.boundaryModels.size(); i++)
	{
		string meshFileName = scene.boundaryModels[i]->meshFile;
		if (FileSystem::isRelativePath(meshFileName))
			meshFileName = FileSystem::normalizePath(scene_path + "/" + scene.boundaryModels[i]->meshFile);

		// check if mesh file has changed
		std::string md5FileName = FileSystem::normalizePath(cachePath + "/" + FileSystem::getFileNameWithExt(meshFileName) + ".md5");
		bool md5 = false;
		if (useCache)
		{
			string md5Str = FileSystem::getFileMD5(meshFileName);
			if (FileSystem::fileExists(md5FileName))
				md5 = FileSystem::checkMD5(md5Str, md5FileName);
		}

    // ---------------------------------------------------------------
		Dynamic3dRigidBody* rb = new Dynamic3dRigidBody();
		rb->setIsAnimated(scene.boundaryModels[i]->isAnimated);
		rb->setIsDynamic(scene.boundaryModels[i]->dynamic);
		TriangleMesh& geo = rb->getGeometry();
		loadObj(meshFileName, geo, scene.boundaryModels[i]->scale);

		std::vector<Vector3r> boundaryParticles;
		if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
		{
			// if a samples file is given, use this one
			if (scene.boundaryModels[i]->samplesFile != "")
			{
				string particleFileName = scene_path + "/" + scene.boundaryModels[i]->samplesFile;
				PartioReaderWriter::readParticles(particleFileName, Vector3r::Zero(), Matrix3r::Identity(), scene.boundaryModels[i]->scale[0], boundaryParticles);
			}
			else		// if no samples file is given, sample the surface model
			{
				// Cache sampling
				std::string mesh_base_path = FileSystem::getFilePath(scene.boundaryModels[i]->meshFile);
				std::string mesh_file_name = FileSystem::getFileName(scene.boundaryModels[i]->meshFile);

				const string resStr = StringTools::real2String(scene.boundaryModels[i]->scale[0]) + "_" + StringTools::real2String(scene.boundaryModels[i]->scale[1]) + "_" + StringTools::real2String(scene.boundaryModels[i]->scale[2]);
				const string modeStr = "_m" + std::to_string(scene.boundaryModels[i]->samplingMode);
				const string particleFileName = FileSystem::normalizePath(cachePath + "/" + mesh_file_name + "_sb_" + StringTools::real2String(scene.particleRadius) + "_" + resStr + modeStr + ".bgeo");

				// check MD5 if cache file is available
				bool foundCacheFile = false;

				if (useCache)
					foundCacheFile = FileSystem::fileExists(particleFileName);

				if (useCache && foundCacheFile && md5)
				{
					PartioReaderWriter::readParticles(particleFileName, Vector3r::Zero(), Matrix3r::Identity(), 1.0, boundaryParticles);
					LOG_INFO << "Loaded cached boundary sampling: " << particleFileName;
				}

				if (!useCache || !foundCacheFile || !md5)
				{
					if (!scene.sim2D)
					{
						const auto samplePoissonDisk = [&]()
						{
							LOG_INFO << "Poisson disk surface sampling of " << meshFileName;
							START_TIMING("Poisson disk sampling");
							PoissonDiskSampling sampling;
							sampling.sampleMesh(geo.numVertices(), geo.getVertices().data(), geo.numFaces(), geo.getFaces().data(), scene.particleRadius, 10, 1, boundaryParticles);
							STOP_TIMING_AVG;
						};
						const auto sampleRegularTriangle = [&]()
						{
							LOG_INFO << "Regular triangle surface sampling of " << meshFileName;
							START_TIMING("Regular triangle sampling");
							RegularTriangleSampling sampling;
							sampling.sampleMesh(geo.numVertices(), geo.getVertices().data(), geo.numFaces(), geo.getFaces().data(), 1.5f * scene.particleRadius, boundaryParticles);
							STOP_TIMING_AVG;
						};
						if (SurfaceSamplingMode::PoissonDisk == scene.boundaryModels[i]->samplingMode)
							samplePoissonDisk();
						else if (SurfaceSamplingMode::RegularTriangle == scene.boundaryModels[i]->samplingMode)
							sampleRegularTriangle();
						else
						{
							LOG_WARN << "Unknown surface sampling method: " << scene.boundaryModels[i]->samplingMode;
							LOG_WARN << "Falling back to:";
							sampleRegularTriangle();
						}
					}
					else
					{
						LOG_INFO << "2D regular sampling of " << meshFileName;
						START_TIMING("2D regular sampling");
						RegularSampling2D sampling;
						sampling.sampleMesh(Matrix3r::Identity(), Vector3r::Zero(),
							geo.numVertices(), geo.getVertices().data(), geo.numFaces(),
							geo.getFaces().data(), 1.75f * scene.particleRadius, boundaryParticles);
						STOP_TIMING_AVG;
					}

					// Cache sampling
					if (useCache && (FileSystem::makeDir(cachePath) == 0))
					{
						LOG_INFO << "Save particle sampling: " << particleFileName;
						PartioReaderWriter::writeParticles(particleFileName, (unsigned int)boundaryParticles.size(), boundaryParticles.data(), nullptr, scene.particleRadius);
					}
				}
			}
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
			BoundaryModel_Akinci2012 *bm = new BoundaryModel_Akinci2012();
			bm->initModel(rb, static_cast<unsigned int>(boundaryParticles.size()), &boundaryParticles[0]);
			sim->addBoundaryModel(bm);
		}
		else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
		{
			BoundaryModel_Koschier2017 *bm = new BoundaryModel_Koschier2017();
			bm->initModel(rb);
			sim->addBoundaryModel(bm);
			SPH::TriangleMesh &mesh = rb->getGeometry();
			m_base->initDensityMap(mesh.getVertices(), mesh.getFaces(), scene.boundaryModels[i], md5, false, bm);
		}
		else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
		{
			BoundaryModel_Bender2019 *bm = new BoundaryModel_Bender2019();
			bm->initModel(rb);
			sim->addBoundaryModel(bm);
			SPH::TriangleMesh &mesh = rb->getGeometry();
			m_base->initVolumeMap(mesh.getVertices(), mesh.getFaces(), scene.boundaryModels[i], md5, false, bm);
		}
		if (useCache && !md5)
			FileSystem::writeMD5File(meshFileName, md5FileName);
		rb->updateMeshTransformation();
	}

	for (unsigned int i = 0; i < scene.articulatedSystems.size();i++)
	{
    // TODO: replace new with smart pointers 
		auto artSystem = new articulatedDynamic3dSystem();
		auto &artSystemData = scene.articulatedSystems[i];
		for(auto index : artSystemData->rigidBodyIndices)
		{
			auto bm = dynamic_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModel(index));
			assert(bm && "Articulated systems are now allowed only for 2012 Akinci boundary handling.\n");
			auto rbo = dynamic_cast<Dynamic3dRigidBody *>(bm->getRigidBodyObject());
			assert(rbo && "3d articulated systems are only allowed for 3d dynamic simulations.");
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


void RigidBody3dBoundarySimulator::deferredInit()
{
	Simulation* sim = Simulation::getCurrent();
  if(sim->useRigidContactSolver())
  {
    m_rigid_contact_solver = std::unique_ptr<RigidContactSolver>(new RigidContactSolver());   
  }
	sim->performNeighborhoodSearchSort();
	if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
	{
		m_base->updateBoundaryParticles(true);
		Simulation::getCurrent()->updateBoundaryVolume();
	}
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
		m_base->updateDMVelocity();
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
		m_base->updateVMVelocity();


}

void RigidBody3dBoundarySimulator::timeStep()
{
  velocityTimeStep();
  positionTimeStep();
}

void RigidBody3dBoundarySimulator::velocityTimeStep()
{
	Real h = TimeManager::getCurrent()->getTimeStepSize();
	Simulation* sim = Simulation::getCurrent();
	const unsigned int nObjects = sim->numberOfBoundaryModels();

  if(sim->useRigidContactSolver()) // for interlinked SPH solver 
  {
    for (unsigned int i = 0; i < nObjects; i++)
    {
      BoundaryModel *bm = sim->getBoundaryModel(i);
      RigidBodyObject *rbo = bm->getRigidBodyObject();
      auto d3drbo = dynamic_cast<Dynamic3dRigidBody *>(rbo);
      if (d3drbo && rbo->isDynamic())
      {
        d3drbo->addGravity();
      }
    }

    m_rigid_contact_solver->solveRigidContact(); 

  }
  else { // Do not use rigid contact solver 
    updateBoundaryForces();

    for (unsigned int i = 0; i < nObjects; i++)
    {
      BoundaryModel* bm = sim->getBoundaryModel(i);
      RigidBodyObject* rbo = bm->getRigidBodyObject();
      auto d3drbo = dynamic_cast<Dynamic3dRigidBody*>(rbo);
      if (d3drbo && rbo->isDynamic())
      {
        if(false == rbo->isAnimated())
          d3drbo->addGravity();
      }
    }
    for (auto &system : articulateSystems)
    {
      system->timeStep();
      //system->checkJointsStatus();
    }
  } // end else
}

void RigidBody3dBoundarySimulator::positionTimeStep()
{
	Real h = TimeManager::getCurrent()->getTimeStepSize();
	Simulation* sim = Simulation::getCurrent();
	const unsigned int nObjects = sim->numberOfBoundaryModels();

  for (unsigned int i = 0; i < nObjects; i++)
  {
    BoundaryModel *bm = sim->getBoundaryModel(i);
    RigidBodyObject *rbo = bm->getRigidBodyObject();
    auto d3drbo = dynamic_cast<Dynamic3dRigidBody *>(rbo);
    if (d3drbo && rbo->isDynamic())
    {
      d3drbo->animate();
    }
  }
	if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
		m_base->updateBoundaryParticles(false);
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
		m_base->updateDMVelocity();
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
		m_base->updateVMVelocity();
}

void RigidBody3dBoundarySimulator::reset()
{
	Simulation* sim = Simulation::getCurrent();
	for (unsigned int i = 0; i < sim->numberOfBoundaryModels(); i++)
	{
		BoundaryModel* bm = sim->getBoundaryModel(i);
		((Dynamic3dRigidBody*)bm->getRigidBodyObject())->reset();
	}

  if(m_rigid_contact_solver != nullptr)
    m_rigid_contact_solver->reset(); 

	sim->performNeighborhoodSearchSort();// Add by ZhehaoLi to fix bottle flip penetration error
	if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Akinci2012)
  {
		m_base->updateBoundaryParticles(true);
		Simulation::getCurrent()->updateBoundaryVolume();// Add by ZhehaoLi to fix bottle flip penetration error
  }
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Koschier2017)
		m_base->updateDMVelocity();
	else if (sim->getBoundaryHandlingMethod() == BoundaryHandlingMethods::Bender2019)
		m_base->updateVMVelocity();
}

RigidContactSolver* RigidBody3dBoundarySimulator::getRigidContactSolver()
{
  return m_rigid_contact_solver.get();
}

void RigidBody3dBoundarySimulator::performNeighborhoodSearchSort()
{
  if(Simulation::getCurrent()->useRigidContactSolver())
    m_rigid_contact_solver->performNeighborhoodSearchSort(); 
}
