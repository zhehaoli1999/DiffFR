#include "BoundaryModel_Akinci2012.h"
#include "SPHKernels.h"
#include <iostream>
#include <vector>
#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/FluidModel.h"
#include "SPlisHSPlasH/RigidBodyObject.h"
#include "SPlisHSPlasH/Utilities/AVX_math.h"
#include "TimeManager.h"
#include "TimeStep.h"
#include "Utilities/Logger.h"
#include "Utilities/ColorfulPrint.h"
#include "NeighborhoodSearch.h"
#include "Simulation.h"
#include "GradientUtils.h"

using namespace SPH;


BoundaryModel_Akinci2012::BoundaryModel_Akinci2012() :
  m_x0(),
  m_x(),
  m_v(),
  m_V(),
  m_pressure()
#ifdef BACKWARD
  ,m_grad_force_to_v(),
  m_grad_force_to_x(),
  m_grad_force_to_quaternion(),
  m_grad_force_to_omega(),

  m_grad_torque_to_omega(),
  m_grad_torque_to_quaternion(),
  m_grad_torque_to_x(),
  m_grad_torque_to_v()
#endif
{
  m_sorted = false;
  m_pointSetIndex = 0;
}

BoundaryModel_Akinci2012::~BoundaryModel_Akinci2012(void)
{
  m_x0.clear();
  m_x.clear();
  m_v.clear();
  m_V.clear();
  m_pressure.clear();
#ifdef BACKWARD
  m_grad_force_to_v.clear();
  m_grad_force_to_x.clear();
  m_grad_force_to_quaternion.clear();
  m_grad_force_to_omega.clear();

  m_grad_torque_to_omega.clear();
  m_grad_torque_to_quaternion.clear();
  m_grad_torque_to_x.clear();
  m_grad_torque_to_v.clear();
#endif
}

void BoundaryModel_Akinci2012::reset_gradient()
{
  auto articulated_system = this->getRigidBodyObject()->getArticulatedSystem();
  if (articulated_system != nullptr)
  {
    articulated_system->reset();
    return;
  }
  else 
  {
    for (int j = 0; j < (int)numberOfParticles(); j++)
      {

        m_grad_force_to_v[j].setZero();
        m_grad_force_to_x[j].setZero();
        m_grad_force_to_quaternion[j].setZero();
        m_grad_force_to_omega[j].setZero();

        m_grad_torque_to_omega[j].setZero();
        m_grad_torque_to_v[j].setZero();
        m_grad_torque_to_x[j].setZero();
        m_grad_torque_to_quaternion[j].setZero();
      }
    grad_net_force_to_vn.setZero();
    grad_net_force_to_xn.setZero();
    grad_net_force_to_omega_n.setZero();
    grad_net_force_to_qn.setZero();

    grad_net_torque_to_omega_n.setZero();
    grad_net_torque_to_vn.setZero();
    grad_net_torque_to_xn.setZero();
    grad_net_torque_to_qn.setZero();
    m_grad_v_to_v0.setIdentity();
    m_grad_v_to_omega0.setZero();
    m_grad_omega_to_omega0.setIdentity();
    m_grad_omega_to_v0.setZero();

    m_grad_quaternion_to_omega0.setZero();
    m_partial_grad_qn_to_omega_n.setZero();
    m_grad_angle_in_radian_to_omega0.setZero();
    m_grad_quaternion_to_v0.setZero();
    m_grad_angle_in_radian_to_v0.setZero();

    m_grad_x_to_v0.setZero();
    m_grad_x_to_omega0.setZero();
  }
}

void BoundaryModel_Akinci2012::reset()
{
  BoundaryModel::reset();

  // Note:
  // positions and velocities are already updated by updateBoundaryParticles
  if (!m_rigidBody->isDynamic() && !m_rigidBody->isAnimated())
  {
    // reset velocities and accelerations
    for (int j = 0; j < (int)numberOfParticles(); j++)
      {
        m_x[j] = m_x0[j];
        m_v[j].setZero();

      }
  }

#ifdef BACKWARD
  for (int j = 0; j < (int)numberOfParticles(); j++)
    {

      m_grad_force_to_v[j].setZero();
      m_grad_force_to_x[j].setZero();
      m_grad_force_to_quaternion[j].setZero();
      m_grad_force_to_omega[j].setZero();

      m_grad_torque_to_omega[j].setZero();
      m_grad_torque_to_v[j].setZero();
      m_grad_torque_to_x[j].setZero();
      m_grad_torque_to_quaternion[j].setZero();
    }
  grad_net_force_to_vn.setZero();
  grad_net_force_to_xn.setZero();
  grad_net_force_to_omega_n.setZero();
  grad_net_force_to_qn.setZero();

  grad_net_torque_to_omega_n.setZero();
  grad_net_torque_to_vn.setZero();
  grad_net_torque_to_xn.setZero();
  grad_net_torque_to_qn.setZero();
  m_grad_v_to_v0.setIdentity();
  m_grad_v_to_omega0.setZero();
  m_grad_omega_to_omega0.setIdentity();
  m_grad_omega_to_v0.setZero();

  m_grad_quaternion_to_omega0.setZero();
  m_partial_grad_qn_to_omega_n.setZero();
  m_grad_angle_in_radian_to_omega0.setZero();
  m_grad_quaternion_to_v0.setZero();
  m_grad_angle_in_radian_to_v0.setZero();

  m_grad_x_to_v0.setZero();
  m_grad_x_to_omega0.setZero();
#endif
}

void BoundaryModel_Akinci2012::computeBoundaryPressureMLS()
{
  Simulation *sim = Simulation::getCurrent();
  const unsigned int nFluids = sim->numberOfFluidModels();
  NeighborhoodSearch *neighborhoodSearch = Simulation::getCurrent()->getNeighborhoodSearch();

  const unsigned int numBoundaryParticles = numberOfParticles();

  #pragma omp parallel default(shared)
  {
    #pragma omp for schedule(static)
    for (int i = 0; i < (int)numBoundaryParticles; i++)
      {
        std::vector<Vector3r> x_bf;  // fluid neighbor positions
        std::vector<Real> vol_bf;
        std::vector<Real> pressure_bf; // fluid neighbor pressures;
        // Traverse all fluid neighbors of one boundary particle to compute pressure
        unsigned int n_fluid_neighbors = 0;
        for (unsigned int pid = 0; pid < sim->numberOfFluidModels(); pid++)
          {
            auto fluid_neighbor = sim->getFluidModel(pid);
            //LOG_INFO << Utilities::RedHead() << "n fluid neighbor[i] = " << sim->numberOfNeighbors(m_pointSetIndex, pid, i)<< Utilities::RedTail();
            for(unsigned int j = 0; j < sim->numberOfNeighbors(m_pointSetIndex, pid, i); j++)
              {
                n_fluid_neighbors ++;
			          const unsigned int neighborIndex = sim->getNeighbor(m_pointSetIndex, pid, i, j); 
                x_bf.push_back(fluid_neighbor->getPosition(neighborIndex));
                pressure_bf.push_back(fluid_neighbor->getPressure(neighborIndex));
                vol_bf.push_back(fluid_neighbor->getVolume(neighborIndex));
              }
          }

        if(n_fluid_neighbors > 0)
        {
          // do MLS here. Ref: Band18 MLS Pressure Extrapolation for the Boundary Handling in DFSPH
          Vector3r x_b = this->getPosition(i); // boundary particle position
          Vector3r sum_x_vol_W_bf = Vector3r::Zero();
          Real sum_vol_W_bf = 0;
          Real sum_p_vol_W_bf = 0;
          for(int bf = 0; bf < x_bf.size(); bf++)
            {
              Real vol_W_bf = vol_bf[bf] * sim->W(x_b - x_bf[bf]);
              sum_vol_W_bf += vol_W_bf;
              sum_x_vol_W_bf += x_bf[bf] * vol_W_bf;
              sum_p_vol_W_bf += pressure_bf[bf] * vol_W_bf;
            }
          Vector3r d_b = sum_x_vol_W_bf / sum_vol_W_bf;

          // Solve for lhs hyperplane parameters of MLS
          Real alpha_b = sum_p_vol_W_bf / sum_vol_W_bf;
          Matrix3r coeff = Matrix3r::Zero();
          Vector3r rhs = Vector3r::Zero();
          for(int bf = 0; bf < x_bf.size(); bf++)
            {
              Vector3r x_bf_bar = x_bf[bf] - d_b;
              Real x = x_bf_bar[0];
              Real y = x_bf_bar[1];
              Real z = x_bf_bar[2];

              Real vol_W_bf = vol_bf[bf] * sim->W(x_b - x_bf[bf]);
              coeff += (Matrix3r() <<
                x*x, x*y, x*z,
                y*x, y*y, y*z,
                z*x, z*y, z*z
              ).finished() * vol_W_bf;

              rhs += x_bf_bar * pressure_bf[bf] * vol_W_bf;
            }

          if(fabs(coeff.determinant()) > 1e-8 )
          {
            Vector3r param = coeff.inverse() * rhs;
            Vector3r x_b_bar = x_b - d_b;
            m_pressure[i] = alpha_b + x_b_bar.dot(param);
          }
          else
            m_pressure[i] = alpha_b;
        }
      }
  }

  //if(sim->isDebug())
  //{
    //for(int i= 0; i < numberOfParticles(); i++)
      //{
        //LOG_INFO << Utilities::RedHead() << "pressure[i] = " << m_pressure[i] << Utilities::RedTail();
      //}

  //}
}

void BoundaryModel_Akinci2012::computeBoundaryVolume()
{
  Simulation *sim = Simulation::getCurrent();
  const unsigned int nFluids = sim->numberOfFluidModels();
  NeighborhoodSearch *neighborhoodSearch = Simulation::getCurrent()->getNeighborhoodSearch();

  const unsigned int numBoundaryParticles = numberOfParticles();

  #pragma omp parallel default(shared)
  {
    #pragma omp for schedule(static)
    for (int i = 0; i < (int)numBoundaryParticles; i++)
      {
        Real delta = sim->W_zero();
        for (unsigned int pid = nFluids; pid < sim->numberOfPointSets(); pid++)
          {
            BoundaryModel_Akinci2012 *bm_neighbor = static_cast<BoundaryModel_Akinci2012*>(sim->getBoundaryModelFromPointSet(pid));
            for (unsigned int j = 0; j < neighborhoodSearch->point_set(m_pointSetIndex).n_neighbors(pid, i); j++)
              {
                const unsigned int neighborIndex = neighborhoodSearch->point_set(m_pointSetIndex).neighbor(pid, i, j);
                delta += sim->W(getPosition(i) - bm_neighbor->getPosition(neighborIndex));
              }
          }
        const Real volume = static_cast<Real>(1.0) / delta;
        m_V[i] = volume;
      }
  }
}

void BoundaryModel_Akinci2012::initModel(RigidBodyObject *rbo, const unsigned int numBoundaryParticles, Vector3r *boundaryParticles)
{
  m_x0.resize(numBoundaryParticles);
  m_x.resize(numBoundaryParticles);
  m_v.resize(numBoundaryParticles);
  m_V.resize(numBoundaryParticles);
  m_pressure.resize(numBoundaryParticles);
#ifdef BACKWARD
  m_grad_force_to_v.resize(numBoundaryParticles); // TODO: actually, we only need to store
  // gradient on the shallow surface of rigidbody?
  m_grad_force_to_omega.resize(numBoundaryParticles);
  m_grad_force_to_quaternion.resize(numBoundaryParticles);
  m_grad_force_to_x.resize(numBoundaryParticles);

  m_grad_torque_to_omega.resize(numBoundaryParticles);
  m_grad_torque_to_v.resize(numBoundaryParticles);
  m_grad_torque_to_x.resize(numBoundaryParticles);
  m_grad_torque_to_quaternion.resize(numBoundaryParticles);

  // --------------------------------------------
  m_grad_x_to_v0.setZero(); // clearGradient;
  m_grad_x_to_omega0.setZero(); // clearGradient;

  m_grad_quaternion_to_omega0.setZero();
  m_partial_grad_qn_to_omega_n.setZero();
  m_grad_angle_in_radian_to_omega0.setZero();
  m_grad_quaternion_to_v0.setZero();
  m_grad_angle_in_radian_to_v0.setZero();

  m_grad_v_to_v0.setIdentity(); // clearGradient;
  m_grad_v_to_omega0.setZero(); // clearGradient;
  m_grad_omega_to_omega0.setIdentity();
  m_grad_omega_to_v0.setZero();

  // --------------------------------------------

  grad_net_force_to_vn.setZero();
  grad_net_force_to_xn.setZero();
  grad_net_force_to_qn.setZero();
  grad_net_force_to_omega_n.setZero();

  grad_net_torque_to_omega_n.setZero();
  grad_net_torque_to_vn.setZero();
  grad_net_torque_to_xn.setZero();
  grad_net_torque_to_qn.setZero();

#endif

  if (rbo->isDynamic())
  {
#ifdef _OPENMP
    const int maxThreads = omp_get_max_threads();
#else
    const int maxThreads = 1;
#endif
    m_forcePerThread.resize(maxThreads, Vector3r::Zero());
    m_torquePerThread.resize(maxThreads, Vector3r::Zero());
    m_forcePerThread_backup.resize(maxThreads, Vector3r::Zero());
    m_torquePerThread_backup.resize(maxThreads, Vector3r::Zero());
  }

  #pragma omp parallel default(shared)
  {
    #pragma omp for schedule(static)
    for (int i = 0; i < (int) numBoundaryParticles; i++)
      {
        m_x0[i] = boundaryParticles[i];
        m_x[i] = boundaryParticles[i];
        m_v[i].setZero();
        m_V[i] = 0.0;
        m_pressure[i] = 0.0;
#ifdef BACKWARD
        m_grad_force_to_v[i].setZero();
        m_grad_force_to_x[i].setZero();
        m_grad_force_to_quaternion[i].setZero();
        m_grad_force_to_omega[i].setZero();

        m_grad_torque_to_omega[i].setZero();
        m_grad_torque_to_v[i].setZero();
        m_grad_torque_to_x[i].setZero();
        m_grad_torque_to_quaternion[i].setZero();
#endif
      }
  }
  m_rigidBody = rbo;

  NeighborhoodSearch *neighborhoodSearch = Simulation::getCurrent()->getNeighborhoodSearch();
  //if(m_rigidBody->isDynamic())
      m_pointSetIndex = neighborhoodSearch->add_point_set(&m_x[0][0], m_x.size(), m_rigidBody->isDynamic() || m_rigidBody->isAnimated(), true, true, this);
      //m_pointSetIndex = neighborhoodSearch->add_point_set(&m_x[0][0], m_x.size(), true, true, true, this);
  //else
      //m_pointSetIndex = neighborhoodSearch->add_point_set(&m_x[0][0], m_x.size(), m_rigidBody->isDynamic() || m_rigidBody->isAnimated(), false, true, this);

  //Vector3r sum_x = Vector3r::Zero();
  //int i;
  //for( i = 0; i < m_x.size(); i++)
  //{
  //sum_x += m_x[i];
  //}
  //std::cout << "!!!!!!!! sum_x = " << (sum_x / i).transpose() << "!!!!!!!" << std::endl;
}

void BoundaryModel_Akinci2012::performNeighborhoodSearchSort()
{
  const unsigned int numPart = numberOfParticles();

  // sort static boundaries only once
  if ((numPart == 0) || (!m_rigidBody->isDynamic() && !m_rigidBody->isAnimated() && m_sorted))
    return;

  NeighborhoodSearch *neighborhoodSearch = Simulation::getCurrent()->getNeighborhoodSearch();

  auto const& d = neighborhoodSearch->point_set(m_pointSetIndex);
  d.sort_field(&m_x0[0]);
  d.sort_field(&m_x[0]);
  d.sort_field(&m_v[0]);
  d.sort_field(&m_V[0]);
  d.sort_field(&m_pressure[0]);
#ifdef BACKWARD
  d.sort_field(&m_grad_force_to_v[0]);
  d.sort_field(&m_grad_force_to_x[0]);
  d.sort_field(&m_grad_force_to_quaternion[0]);
  d.sort_field(&m_grad_force_to_omega[0]);

  d.sort_field(&m_grad_torque_to_omega[0]);
  d.sort_field(&m_grad_torque_to_quaternion[0]);
  d.sort_field(&m_grad_torque_to_x[0]);
  d.sort_field(&m_grad_torque_to_v[0]);
#endif
  m_sorted = true;
}

void SPH::BoundaryModel_Akinci2012::saveState(BinaryFileWriter &binWriter)
{
  binWriter.write(m_sorted);
  binWriter.write(m_pointSetIndex);
}

void SPH::BoundaryModel_Akinci2012::loadState(BinaryFileReader &binReader)
{
  binReader.read(m_sorted);
  binReader.read(m_pointSetIndex);
}

void SPH::BoundaryModel_Akinci2012::resize(const unsigned int numBoundaryParticles)
{
  m_x0.resize(numBoundaryParticles);
  m_x.resize(numBoundaryParticles);
  m_v.resize(numBoundaryParticles);
  m_V.resize(numBoundaryParticles);
  m_pressure.resize(numBoundaryParticles);

#ifdef BACKWARD
  m_grad_force_to_v.resize(numBoundaryParticles);
  m_grad_force_to_x.resize(numBoundaryParticles);
  m_grad_force_to_quaternion.resize(numBoundaryParticles);
  m_grad_force_to_omega.resize(numBoundaryParticles);

  m_grad_torque_to_omega.resize(numBoundaryParticles);
  m_grad_torque_to_quaternion.resize(numBoundaryParticles);
  m_grad_torque_to_x.resize(numBoundaryParticles);
  m_grad_torque_to_v.resize(numBoundaryParticles);
#endif

}

#ifdef BACKWARD
void SPH::BoundaryModel_Akinci2012::accumulate_and_reset_gradient()
{
  this->grad_net_force_to_vn.setZero();
  this->grad_net_force_to_xn.setZero();
  this->grad_net_force_to_omega_n.setZero();
  this->grad_net_force_to_qn.setZero();

  this->grad_net_torque_to_omega_n.setZero();
  this->grad_net_torque_to_qn.setZero();
  this->grad_net_torque_to_vn.setZero();
  this->grad_net_torque_to_xn.setZero();

  const unsigned int numPart = numberOfParticles();
  if(numPart > 0)
  {
    for (int i = 0; i < (int) numPart; i++)
      {
        auto vol = getVolume(i);
        this->grad_net_force_to_vn += m_grad_force_to_v[i];
        this->grad_net_force_to_xn += m_grad_force_to_x[i];
        this->grad_net_force_to_omega_n += m_grad_force_to_omega[i];
        this->grad_net_force_to_qn += m_grad_force_to_quaternion[i];

        this->grad_net_torque_to_omega_n += m_grad_torque_to_omega[i];
        this->grad_net_torque_to_qn += m_grad_torque_to_quaternion[i];
        this->grad_net_torque_to_vn += m_grad_torque_to_v[i];
        this->grad_net_torque_to_xn += m_grad_torque_to_x[i];

        m_grad_force_to_v[i].setZero();
        m_grad_force_to_x[i].setZero();
        m_grad_force_to_omega[i].setZero();
        m_grad_force_to_quaternion[i].setZero();

        m_grad_torque_to_omega[i].setZero();
        m_grad_torque_to_v[i].setZero();
        m_grad_torque_to_x[i].setZero();
        m_grad_torque_to_quaternion[i].setZero();
      }

    //std::cout << Utilities::GreenHead() << "total_grad_force_to_x_rb = \n"
    //<< total_grad_force_to_x_rb << Utilities::GreenTail() << std::endl;
    //std::cout << Utilities::GreenHead() << "total_grad_force_to_v_rb = \n"
    //<< total_grad_force_to_v_rb << Utilities::GreenTail() << std::endl;
  }
}

// compute
Matrix3r BoundaryModel_Akinci2012::compute_grad_Rv_to_omega0(Vector3r v)
{
  Matrix34r grad_Rv_to_q = get_grad_Rqp_to_q(getRigidBodyObject()->getRotation(), v);
  return grad_Rv_to_q * m_grad_quaternion_to_omega0;
}

Matrix3r BoundaryModel_Akinci2012::compute_grad_RTv_to_omega0(Vector3r v)
{
  Matrix34r grad_RTv_to_q = get_grad_RqTp_to_q(getRigidBodyObject()->getRotation(), v);
  return grad_RTv_to_q * m_grad_quaternion_to_omega0;
}
// -------------------------------------------------------------------------------------

Matrix3r BoundaryModel_Akinci2012::compute_grad_Rv_to_v0(Vector3r v)
{
  Matrix34r grad_Rv_to_q = get_grad_Rqp_to_q(getRigidBodyObject()->getRotation(), v);
  return grad_Rv_to_q * m_grad_quaternion_to_v0;
}

Matrix3r BoundaryModel_Akinci2012::compute_grad_RTv_to_v0(Vector3r v)
{
  Matrix34r grad_RTv_to_q = get_grad_RqTp_to_q(getRigidBodyObject()->getRotation(), v);
  return grad_RTv_to_q * m_grad_quaternion_to_v0;
}

// -------------------------------------------------------------------------------------

Matrix3r BoundaryModel_Akinci2012::compute_grad_inertia_v_to_omega0(Vector3r v)
{
  Matrix3r inertia_0 = getRigidBodyObject()->getInertiaTensor0();
  Matrix3r R = getRigidBodyObject()->getRotation().toRotationMatrix();
  return ( compute_grad_Rv_to_omega0(inertia_0 * R.transpose() * v)
  +  R*inertia_0* compute_grad_RTv_to_omega0(v)  );
}

Matrix3r BoundaryModel_Akinci2012::compute_grad_inertia_v_to_v0(Vector3r v)
{
  Matrix3r inertia_0 = getRigidBodyObject()->getInertiaTensor0();
  Matrix3r R = getRigidBodyObject()->getRotation().toRotationMatrix();
  return ( compute_grad_Rv_to_v0(inertia_0 * R.transpose() * v)
  +  R*inertia_0* compute_grad_RTv_to_v0(v)  );
}

void SPH::BoundaryModel_Akinci2012::perform_chain_rule(const Real timeStep, const bool optimize_rotation)
{

  auto articulated_system = this->getRigidBodyObject()->getArticulatedSystem();
  if (articulated_system != nullptr)
  {
    articulated_system->perform_chain_rule(timeStep);
    //system->get_bm_grad_information(this, m_grad_v_to_v0, m_grad_v_to_omega0, m_grad_omega_to_v0, m_grad_omega_to_omega0,
    //m_grad_x_to_v0, m_grad_x_to_omega0, m_grad_quaternion_to_v0, m_grad_quaternion_to_omega0);
    return;
  }
  auto sim = Simulation::getCurrent();
  GradientMode gradient_mode = static_cast<GradientMode>(sim->getGradientMode());
  RigidBodyMode rb_mode = static_cast<RigidBodyMode>(sim->getRigidBodyMode());

  const Real dt = timeStep;
  const Real mass = getRigidBodyObject()->getMass();
  const Real invMass = static_cast<Real>(1.0) / (mass + 1e-10);
  Matrix3r inertia = getRigidBodyObject()->getInertiaTensorWorld();
  Matrix3r inv_inertia = getRigidBodyObject()->getInertiaTensorInverseWorld();
  Matrix3r R = getRigidBodyObject()->getRotation().toRotationMatrix();
  auto omega = getRigidBodyObject()->getAngularVelocity();

  auto I = Matrix3r::Identity();
  Vector3r temp_v = Vector3r::Zero();

  if(gradient_mode == GradientMode::Complete)
  {
    auto matrix2norm = [] (Eigen::MatrixXf M)
      {
        Eigen::MatrixXf Mt = M.adjoint();
        Eigen::MatrixXf MtM = Mt * M;
        Eigen::VectorXf eigenvalues = MtM.eigenvalues().real();
        double max_eigenvalue = eigenvalues.maxCoeff();
        return std::sqrt(max_eigenvalue);
      };

    Matrix3r grad_net_force_to_v0 = grad_net_force_to_xn * m_grad_x_to_v0
      + grad_net_force_to_vn * m_grad_v_to_v0
      + grad_net_force_to_qn * m_grad_quaternion_to_v0
      + grad_net_force_to_omega_n * m_grad_omega_to_v0;

    Matrix3r grad_net_force_to_omega0 = grad_net_force_to_xn * m_grad_x_to_omega0
      + grad_net_force_to_vn * m_grad_v_to_omega0
      + grad_net_force_to_qn * m_grad_quaternion_to_omega0
      + grad_net_force_to_omega_n * m_grad_omega_to_omega0;

    //Matrix3r grad_v_coeff = ( I + dt * m_grad_v_to_v0.inverse() * invMass * grad_net_force_to_v0);
    //m_grad_v_to_v0 =  m_grad_v_to_v0 * (grad_v_coeff / matrix2norm(grad_v_coeff));

    m_grad_v_to_v0 = m_grad_v_to_v0 + dt * invMass * grad_net_force_to_v0;
    m_grad_v_to_omega0 =  m_grad_v_to_omega0 + dt * invMass * grad_net_force_to_omega0;

    //if(m_grad_x_to_v0.norm() > 1e-6)
    //{
    //Matrix3r grad_x_coeff = (I + dt * m_grad_x_to_v0.inverse() * m_grad_v_to_v0);
    //m_grad_x_to_v0 = m_grad_x_to_v0 * (grad_x_coeff / matrix2norm(grad_x_coeff));
    //}
    //else
    //m_grad_x_to_v0 =  m_grad_x_to_v0 + dt * m_grad_v_to_v0;

    m_grad_x_to_v0 =  m_grad_x_to_v0 + dt * m_grad_v_to_v0;
    m_grad_x_to_omega0 =  m_grad_x_to_omega0 + dt * m_grad_v_to_omega0;

    Vector3r force, torque;
    this->getForceAndTorque(force, torque);

    Matrix3r grad_net_torque_to_omega0 = grad_net_torque_to_xn * m_grad_x_to_omega0
      + grad_net_torque_to_vn * m_grad_v_to_omega0
      + grad_net_torque_to_qn * m_grad_quaternion_to_omega0
      + grad_net_torque_to_omega_n * m_grad_omega_to_omega0;

    Matrix3r grad_net_torque_to_v0 = grad_net_torque_to_xn * m_grad_x_to_v0
      + grad_net_torque_to_vn * m_grad_v_to_v0
      + grad_net_torque_to_qn * m_grad_quaternion_to_v0
      + grad_net_torque_to_omega_n * m_grad_omega_to_v0;

    Matrix3r grad_Tau_to_omega0= Matrix3r::Zero();
    Matrix3r grad_Tau_to_v0 = Matrix3r::Zero();
    if(rb_mode == RigidBodyMode::WithGyroscopic)
    {
      temp_v = (inertia * omega).cross(omega) + torque; // TODO: duplicate computation here.

      Matrix3r grad_L_cross_omega_to_omega0 = skewMatrix(inertia * omega) * m_grad_omega_to_omega0
        + skewMatrix(omega).transpose() * (inertia * m_grad_omega_to_omega0 + compute_grad_inertia_v_to_omega0(omega) );

      Matrix3r grad_L_cross_omega_to_v0 = skewMatrix(inertia * omega) * m_grad_omega_to_v0
        + skewMatrix(omega).transpose() * (inertia * m_grad_omega_to_v0 + compute_grad_inertia_v_to_v0(omega) );

      grad_Tau_to_omega0 = (grad_L_cross_omega_to_omega0 + grad_net_torque_to_omega0);
      grad_Tau_to_v0 = (grad_L_cross_omega_to_v0 + grad_net_torque_to_v0);
    }
  else
    {
      temp_v = torque; // TODO: duplicate computation here.
      grad_Tau_to_omega0 = (grad_net_torque_to_omega0);
      grad_Tau_to_v0 = (grad_net_torque_to_v0);
    }
    Matrix3r grad_inv_inertia_v_to_omega0 = inv_inertia * compute_grad_inertia_v_to_omega0(inv_inertia * temp_v);
    Matrix3r grad_inv_inertia_v_to_v0 = inv_inertia * compute_grad_inertia_v_to_v0(inv_inertia * temp_v);

    m_grad_omega_to_omega0 = m_grad_omega_to_omega0 + dt * (grad_inv_inertia_v_to_omega0
      + inv_inertia * grad_Tau_to_omega0);
    m_grad_omega_to_v0 = m_grad_omega_to_v0 + dt * (grad_inv_inertia_v_to_v0
      + inv_inertia * grad_Tau_to_v0);
    //Matrix3r m_grad_omega_coeff = (I + dt * m_grad_omega_to_omega0.inverse() *(grad_inv_inertia_v_to_omega0
    //+ inv_inertia * (grad_L_cross_omega_to_omega0 + grad_net_torque_to_omega0) ) );
    //m_grad_omega_to_omega0 = m_grad_omega_to_omega0 * (m_grad_omega_coeff / matrix2norm(m_grad_omega_coeff));

    if(sim->isDebug())
    {
      LOG_INFO << Utilities::RedHead() <<  "gradient_mode = \n" << (int)gradient_mode << Utilities::YellowTail() << "\n";
      //LOG_INFO << Utilities::YellowHead() <<  "net force = " << force.transpose() << Utilities::YellowTail();
      //LOG_INFO << Utilities::YellowHead() <<  "net torque = " << torque.transpose() << Utilities::YellowTail() << "\n";

      //LOG_INFO << Utilities::CyanHead() <<  "grad_v_coeff = \n" << grad_v_coeff << Utilities::YellowTail() << "\n";
      //LOG_INFO << Utilities::CyanHead() <<  "grad_v_coeff / 2-norm = \n" << grad_v_coeff / matrix2norm(grad_v_coeff) << Utilities::YellowTail() << "\n";
      //LOG_INFO << Utilities::CyanHead() <<  " grad_x_coeff = \n" << grad_x_coeff << Utilities::YellowTail() << "\n";

      LOG_INFO << Utilities::GreenHead() <<  "grad_net_force_to_vn = \n" << grad_net_force_to_vn<< Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::GreenHead() <<  "grad_net_force_to_xn = \n" << grad_net_force_to_xn << Utilities::YellowTail() << "\n";
      //LOG_INFO << Utilities::GreenHead() <<  "grad_net_force_to_qn = \n" << grad_net_force_to_qn << Utilities::YellowTail() << "\n";
      //LOG_INFO << Utilities::GreenHead() <<  "grad_net_force_to_omega_n = \n" << grad_net_force_to_omega_n << Utilities::YellowTail() << "\n";

      LOG_INFO << Utilities::CyanHead() <<  "grad_net_force_to_v0 = \n" << grad_net_force_to_v0 << Utilities::YellowTail() << "\n";
      //LOG_INFO << Utilities::CyanHead() <<  "grad_net_torque_to_omega0 = \n" << grad_net_torque_to_omega0 << Utilities::YellowTail() << "\n";


    }
  }
  // =========================================================================================
else if(gradient_mode == GradientMode::RigidGradOnly)
  {
    // Note: no need to add the two lines below
    //m_grad_v_to_v0 =  m_grad_v_to_v0;
    //m_grad_v_to_omega0 = m_grad_v_to_omega0;

    m_grad_x_to_v0 =  m_grad_x_to_v0 + dt * m_grad_v_to_v0;
    m_grad_x_to_omega0 =  m_grad_x_to_omega0 + dt * m_grad_v_to_omega0;

    if(rb_mode == RigidBodyMode::WithGyroscopic)
    {
      temp_v = (inertia * omega).cross(omega); // TODO: duplicate computation here.
      Matrix3r grad_inv_inertia_v_to_omega0 = inv_inertia * compute_grad_inertia_v_to_omega0(inv_inertia * temp_v);
      Matrix3r grad_inv_inertia_v_to_v0 = inv_inertia * compute_grad_inertia_v_to_v0(inv_inertia * temp_v);

      Matrix3r grad_L_cross_omega_to_omega0 = skewMatrix(inertia * omega) * m_grad_omega_to_omega0
        + skewMatrix(omega).transpose() * (inertia * m_grad_omega_to_omega0 + compute_grad_inertia_v_to_omega0(omega) );
      Matrix3r grad_L_cross_omega_to_v0 = skewMatrix(inertia * omega) * m_grad_omega_to_v0
        + skewMatrix(omega).transpose() * (inertia * m_grad_omega_to_v0 + compute_grad_inertia_v_to_v0(omega) );

      m_grad_omega_to_omega0 = m_grad_omega_to_omega0 + dt * (grad_inv_inertia_v_to_omega0
        + inv_inertia * (grad_L_cross_omega_to_omega0) );
      m_grad_omega_to_v0 = m_grad_omega_to_v0 + dt * (grad_inv_inertia_v_to_v0
        + inv_inertia * (grad_L_cross_omega_to_v0) );
    }
  else {
      // Note: no need to add the two lines below
      //m_grad_omega_to_omega0 = m_grad_omega_to_omega0;
      //m_grad_omega_to_v0 = m_grad_omega_to_v0;
    }

    if(sim->isDebug())
    {
      LOG_INFO << Utilities::RedHead() <<  "gradient_mode = \n" << (int)gradient_mode << Utilities::YellowTail() << "\n";
    }
  }
  // =========================================================================================
else if(gradient_mode == GradientMode::Incomplete)
  {
    Matrix3r partial_grad_xn_to_vn = dt * Matrix3r::Identity();
    //Matrix3r grad_net_force_to_v0 = grad_net_force_to_xn * partial_grad_xn_to_vn * m_grad_v_to_v0
      //+ grad_net_force_to_vn * m_grad_v_to_v0
      //+ grad_net_force_to_qn * m_partial_grad_qn_to_omega_n * m_grad_omega_to_v0
      //+ grad_net_force_to_omega_n * m_grad_omega_to_v0;
    Matrix3r grad_net_force_to_v0 = 
        grad_net_force_to_vn * m_grad_v_to_v0
      + grad_net_force_to_omega_n * m_grad_omega_to_v0;

    //Matrix3r grad_net_force_to_omega0 = grad_net_force_to_xn * partial_grad_xn_to_vn * m_grad_v_to_omega0
      //+ grad_net_force_to_vn * m_grad_v_to_omega0
      //+ grad_net_force_to_qn * m_partial_grad_qn_to_omega_n * m_grad_omega_to_omega0
      //+ grad_net_force_to_omega_n * m_grad_omega_to_omega0;
    Matrix3r grad_net_force_to_omega0 = 
        grad_net_force_to_vn * m_grad_v_to_omega0
      + grad_net_force_to_omega_n * m_grad_omega_to_omega0;

    m_grad_v_to_v0 =  m_grad_v_to_v0 + dt * invMass * grad_net_force_to_v0;
    m_grad_v_to_omega0 =  m_grad_v_to_omega0 + dt * invMass * grad_net_force_to_omega0;

    m_grad_x_to_v0 =  m_grad_x_to_v0 + dt * m_grad_v_to_v0;
    m_grad_x_to_omega0 =  m_grad_x_to_omega0 + dt * m_grad_v_to_omega0;

    //Matrix3r grad_net_torque_to_omega0 = grad_net_torque_to_xn * partial_grad_xn_to_vn * m_grad_v_to_omega0
      //+ grad_net_torque_to_vn * m_grad_v_to_omega0
      //+ grad_net_torque_to_qn * m_partial_grad_qn_to_omega_n * m_grad_omega_to_omega0
      //+ grad_net_torque_to_omega_n * m_grad_omega_to_omega0;

    //Matrix3r grad_net_torque_to_v0 = grad_net_torque_to_xn * partial_grad_xn_to_vn * m_grad_v_to_v0
      //+ grad_net_torque_to_vn * m_grad_v_to_v0
      //+ grad_net_torque_to_qn * m_partial_grad_qn_to_omega_n* m_grad_omega_to_v0
      //+ grad_net_torque_to_omega_n * m_grad_omega_to_v0;

    Matrix3r grad_net_torque_to_omega0 = 
        grad_net_torque_to_vn * m_grad_v_to_omega0
      + grad_net_torque_to_omega_n * m_grad_omega_to_omega0;

    Matrix3r grad_net_torque_to_v0 = 
        grad_net_torque_to_vn * m_grad_v_to_v0
      + grad_net_torque_to_omega_n * m_grad_omega_to_v0;

    Vector3r force, torque;
    this->getForceAndTorque(force, torque);

    Matrix3r grad_Tau_to_omega0= Matrix3r::Zero();
    Matrix3r grad_Tau_to_v0 = Matrix3r::Zero();
    if(rb_mode == RigidBodyMode::WithGyroscopic)
    {
      temp_v = (inertia * omega).cross(omega) + torque; // TODO: duplicate computation here.

      Matrix3r grad_L_cross_omega_to_omega0 = skewMatrix(inertia * omega) * m_grad_omega_to_omega0
        + skewMatrix(omega).transpose() * (inertia * m_grad_omega_to_omega0 + compute_grad_inertia_v_to_omega0(omega) );

      Matrix3r grad_L_cross_omega_to_v0 = skewMatrix(inertia * omega) * m_grad_omega_to_v0
        + skewMatrix(omega).transpose() * (inertia * m_grad_omega_to_v0 + compute_grad_inertia_v_to_v0(omega) );

      if(sim->isDebug())
      {
        LOG_INFO << Utilities::CyanHead() <<  "grad_L_cross_omega_to_omega0 = " << grad_L_cross_omega_to_omega0 << Utilities::YellowTail() << "\n";
        LOG_INFO << Utilities::CyanHead() <<  "grad_L_cross_omega_to_v0 = " << grad_L_cross_omega_to_v0 << Utilities::YellowTail() << "\n";
      }

      grad_Tau_to_omega0 = (grad_L_cross_omega_to_omega0 + grad_net_torque_to_omega0);
      grad_Tau_to_v0 = (grad_L_cross_omega_to_v0 + grad_net_torque_to_v0);
    }
  else
    {
      temp_v = torque; // TODO: duplicate computation here.
      grad_Tau_to_omega0 = (grad_net_torque_to_omega0);
      grad_Tau_to_v0 = (grad_net_torque_to_v0);
    }
    Matrix3r grad_inv_inertia_v_to_omega0 = inv_inertia * compute_grad_inertia_v_to_omega0(inv_inertia * temp_v);
    Matrix3r grad_inv_inertia_v_to_v0 = inv_inertia * compute_grad_inertia_v_to_v0(inv_inertia * temp_v);

    m_grad_omega_to_omega0 = m_grad_omega_to_omega0 + dt * (grad_inv_inertia_v_to_omega0
      + inv_inertia * grad_Tau_to_omega0);
    m_grad_omega_to_v0 = m_grad_omega_to_v0 + dt * (grad_inv_inertia_v_to_v0
      + inv_inertia * grad_Tau_to_v0);

    if(sim->isDebug())
    {
      LOG_INFO << Utilities::RedHead() <<  "gradient_mode = \n" << (int)gradient_mode << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::YellowHead() <<  "net force = " << force.transpose() << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::YellowHead() <<  "net torque = " << torque.transpose() << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::CyanHead() <<  "dt = " << dt << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::CyanHead() <<  "inv_mass = " << invMass << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::CyanHead() <<  "omega = " << omega << Utilities::YellowTail() << "\n";

      LOG_INFO << Utilities::CyanHead() <<  "grad_inv_inertia_v_to_omega0 = " << grad_inv_inertia_v_to_omega0 << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::CyanHead() <<  "grad_inv_inertia_v_to_v0 = " << grad_inv_inertia_v_to_v0 << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::CyanHead() <<  "grad_Tau_to_omega0 = " << grad_Tau_to_omega0 << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::CyanHead() <<  "grad_Tau_to_v0 = " << grad_Tau_to_v0 << Utilities::YellowTail() << "\n";

      LOG_INFO << Utilities::GreenHead() <<  "grad_net_force_to_vn = \n" << grad_net_force_to_vn<< Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::GreenHead() <<  "grad_net_force_to_omega_n = \n" << grad_net_force_to_omega_n << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::CyanHead() <<  "grad_net_torque_to_vn = \n" << grad_net_torque_to_vn<< Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::CyanHead() <<  "grad_net_torque_to_omega_n = \n" << grad_net_torque_to_omega_n << Utilities::YellowTail() << "\n";

      //LOG_INFO << Utilities::CyanHead() <<  "grad_net_force_to_v0 = \n" << grad_net_force_to_v0 << Utilities::YellowTail() << "\n";
      //LOG_INFO << Utilities::CyanHead() <<  "grad_net_torque_to_omega0 = \n" << grad_net_torque_to_omega0 << Utilities::YellowTail() << "\n";
      //
      LOG_INFO << Utilities::YellowHead() <<  "m_grad_v_to_v0 = \n" << m_grad_v_to_v0 << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::YellowHead() <<  "m_grad_v_to_omega0 = \n" << m_grad_v_to_omega0 << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::YellowHead() <<  "m_grad_omega_to_omega0 = \n" << m_grad_omega_to_omega0 << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::YellowHead() <<  "m_grad_omega_to_v0 = \n" << m_grad_omega_to_v0 << Utilities::YellowTail() << "\n";

      LOG_INFO << Utilities::YellowHead() <<  "m_grad_x_to_v0 = \n" << m_grad_x_to_v0 << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::YellowHead() <<  "m_grad_x_to_omega0 = \n" << m_grad_x_to_omega0 << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::YellowHead() <<  "m_grad_q_to_v0 = \n" << m_grad_quaternion_to_v0 << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::YellowHead() <<  "m_grad_q_to_omega0 = \n" << m_grad_quaternion_to_omega0 << Utilities::YellowTail() << "\n";
    }
  }


  if(optimize_rotation)
  {
    // ================= grad to rotation ==========================
    // Ref: https://math.stackexchange.com/questions/3519677/derivative-of-qua%20ternion-multiplication-with-angular-velocity
    auto q = getRigidBodyObject()->getRotation();
    Vector3r new_omega = omega + dt * inv_inertia * temp_v;

    //auto p = omega_q_product;
    auto p = Quaternionr(0., new_omega[0], new_omega[1], new_omega[2]);
    Quaternionr new_q = q;
    new_q.coeffs() += dt * 0.5 * (p * new_q).coeffs();
    auto new_qn = new_q.normalized();
    auto new_qnv = Vector4r(new_qn.w(), new_qn.x(), new_qn.y(), new_qn.z());

    Matrix4r grad_p_q_product_to_q = get_grad_p_q_product_to_q(p);
    Matrix43r grad_p_q_product_to_omega = get_grad_omega_q_product_to_omega(q);

    Matrix4r grad_normalized_q_to_q =(Matrix4r::Identity() - new_qnv * new_qnv.transpose()) / new_q.norm();

    m_partial_grad_qn_to_omega_n = grad_normalized_q_to_q *
      (dt / 2. * (grad_p_q_product_to_omega ) );

    //if(gradient_mode == GradientMode::Complete && (m_grad_quaternion_to_omega0.norm() > 1e-6))
    //{
    //m_grad_quaternion_to_omega0 = grad_normalized_q_to_q * (m_grad_quaternion_to_omega0 * (Matrix3r::Identity() +
    //dt / 2. * m_grad_quaternion_to_omega0.inverse()
    //* (grad_p_q_product_to_omega * m_grad_omega_to_omega0 // 4x3 * 3x3
    //+ grad_p_q_product_to_q * m_grad_quaternion_to_omega0) ).normalized() ); // 4x4 * 4*3
    //}
    //else
    //{
    m_grad_quaternion_to_omega0 = grad_normalized_q_to_q * (m_grad_quaternion_to_omega0 + dt / 2.
      * (grad_p_q_product_to_omega * m_grad_omega_to_omega0 // 4x3 * 3x3
      + grad_p_q_product_to_q * m_grad_quaternion_to_omega0) ); // 4x4 * 4*3
    //}
    //m_grad_quaternion_to_omega0.normalize();

    //LOG_INFO << Utilities::GreenHead() <<  "grad_p_q_product_to_q = \n" << grad_p_q_product_to_q << Utilities::YellowTail() << "\n";
    //LOG_INFO << Utilities::GreenHead() <<  "grad_p_q_product_to_omega = \n" << grad_p_q_product_to_omega << Utilities::YellowTail() << "\n";
    //LOG_INFO << Utilities::GreenHead() <<  "grad_normalized_q_to_q = \n" << grad_normalized_q_to_q << Utilities::YellowTail() << "\n";

    m_grad_quaternion_to_v0 = grad_normalized_q_to_q * (m_grad_quaternion_to_v0 + dt / 2.
      * ( grad_p_q_product_to_omega * m_grad_omega_to_v0 // 4x3 * 3x3
      + grad_p_q_product_to_q * m_grad_quaternion_to_v0)); // 4x4 * 4*3

    // =============================================

    if(sim->isDebug())
    {
      //LOG_INFO << Utilities::RedHead() <<  "grad_net_force_to_xn = \n" << grad_net_force_to_xn << Utilities::YellowTail() << "\n";
      //LOG_INFO << Utilities::RedHead() <<  "grad_net_force_to_vn = \n" << grad_net_force_to_vn << Utilities::YellowTail() << "\n";
      //LOG_INFO << Utilities::RedHead() <<  "grad_net_force_to_qn = \n" << grad_net_force_to_qn << Utilities::YellowTail() << "\n";
      //LOG_INFO << Utilities::RedHead() <<  "grad_net_force_to_omega_n = \n" << grad_net_force_to_omega_n << Utilities::YellowTail() << "\n";
      //
      //LOG_INFO << Utilities::YellowHead() <<  "m_grad_v_to_v0 = \n" << m_grad_v_to_v0 << Utilities::YellowTail() << "\n";
      //LOG_INFO << Utilities::YellowHead() <<  "m_grad_x_to_v0 = \n" << m_grad_x_to_v0 << Utilities::YellowTail() << "\n";

      //LOG_INFO << Utilities::GreenHead() <<  "m_grad_omega_to_omega0 = \n" << m_grad_omega_to_omega0 << Utilities::YellowTail() << "\n";
      //LOG_INFO << Utilities::GreenHead() <<  "m_grad_quaternion_to_omega0 = \n" << m_grad_quaternion_to_omega0 << Utilities::YellowTail() << "\n";
      LOG_INFO << Utilities::GreenHead() <<  " ==============================";
    }
  }
}
#endif


//Matrix3r SPH::BoundaryModel_Akinci2012::get_grad_x_to_v0_another_bm(unsigned int index)
//{
//Simulation *sim = Simulation::getCurrent();
//auto bm = dynamic_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModel(index));
//auto system = m_rigidBody->getSystem();
//if (system && bm)
//{
//return system->get_grad_x_to_v0(this, bm);
//}
//else
//{
//return Matrix3r::Zero();
//}
//}
Matrix3r SPH::BoundaryModel_Akinci2012::get_grad_x_to_omega0_another_bm(unsigned int index)
{
  Simulation *sim = Simulation::getCurrent();
  auto bm = dynamic_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModel(index));
  auto system = m_rigidBody->getArticulatedSystem();
  if (system && bm)
  {
    return system->get_grad_x_to_omega0(this, bm);
  }
else
  {
    return Matrix3r::Zero();
  }
}
Matrix3r SPH::BoundaryModel_Akinci2012::get_grad_x_to_v0_another_bm(unsigned int index)
{
  Simulation *sim = Simulation::getCurrent();
  auto bm = dynamic_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModel(index));
  auto system = m_rigidBody->getArticulatedSystem();
  if (system && bm)
  {
    return system->get_grad_x_to_v0(this, bm);
  }
else
  {
    return Matrix3r::Zero();
  }
}

Matrix43r SPH::BoundaryModel_Akinci2012::get_grad_quaternion_to_v0_another_bm(unsigned int index)
{
  Simulation *sim = Simulation::getCurrent();
  auto bm = dynamic_cast<BoundaryModel_Akinci2012 *>(sim->getBoundaryModel(index));
  auto system = m_rigidBody->getArticulatedSystem();
  if (system && bm)
  {
    return system->get_grad_quaternion_to_v0(this, bm);
  }
else
  {
    return Matrix43r::Zero();
  }
}


void BoundaryModel_Akinci2012::update_rigid_body_gradient_manager()
{
  Simulation *sim = Simulation::getCurrent();
  auto rb_grad_manager = sim->getRigidBodyGradientManager();

  const unsigned R_index = this->getPointSetIndex() - sim->numberOfFluidModels(); 
  rb_grad_manager->get_grad_net_force_to_vn(R_index, R_index) = grad_net_force_to_vn;
  rb_grad_manager->get_grad_net_force_to_xn(R_index, R_index) = grad_net_force_to_xn;
  rb_grad_manager->get_grad_net_force_to_omega_n(R_index, R_index) = grad_net_force_to_omega_n;
  rb_grad_manager->get_grad_net_force_to_qn(R_index, R_index) = grad_net_force_to_qn;

  rb_grad_manager->get_grad_net_torque_to_vn(R_index, R_index) = grad_net_torque_to_vn;
  rb_grad_manager->get_grad_net_torque_to_xn(R_index, R_index) = grad_net_torque_to_xn;
  rb_grad_manager->get_grad_net_torque_to_omega_n(R_index, R_index) = grad_net_torque_to_omega_n;
  rb_grad_manager->get_grad_net_torque_to_qn(R_index, R_index) = grad_net_torque_to_qn;
};
