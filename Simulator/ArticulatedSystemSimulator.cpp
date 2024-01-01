#include "ArticulatedSystemSimulator.h"
#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/RigidBodyObject.h"
#include "SPlisHSPlasH/TimeManager.h"
#include "Simulator/Dynamic2dBoundarySimulator.h"
#include "Utilities/ColorfulPrint.h"
#include "Utilities/Logger.h"
#include "SPlisHSPlasH/GradientUtils.h"
#include "SPlisHSPlasH/Simulation.h"

using namespace SPH; 
using namespace Utilities; 
const bool isArtSystemDebug = false;

ArticulatedDynamicSystemBase::ArticulatedDynamicSystemBase()
: n_actuator_joint(0),
  m_useLagrangian(false) 
{
}

void ArticulatedDynamicSystemBase::addJoint(unsigned int index1, unsigned int index2, 
                                            const Vector3r &position, bool isActuator)
{
	size_t bodyNum = bodies.size();
	if (index1 < bodyNum && index2 < bodyNum)
	{
		auto &rbo1 = bodies[index1];
		auto &rbo2 = bodies[index2];
		joints.push_back(Balljoint({index1, index2,
									rbo1->getRotation0().toRotationMatrix().transpose() * (position - rbo1->getPosition0()),
									rbo2->getRotation0().toRotationMatrix().transpose() * (position - rbo2->getPosition0()),
                  isActuator}));
    if(isActuator)
      n_actuator_joint ++; 
	}
}

void ArticulatedDynamicSystemBase::init()
{
  size_t bodyNum = bodies.size();
  size_t massMatrixSize = 6 * bodyNum; 
  massMatrix.resize(massMatrixSize, massMatrixSize);
  massMatrix.setZero();

  for (size_t i = 0; i < bodyNum; i++)
  {
    auto &rbo = bodies[i];      
    massMatrix.block(6 * i, 6*i, 3, 3) = Matrix3r::Identity() * rbo->getMass();
		massMatrix.block(6 * i + 3, 6 * i + 3, 3, 3) = rbo->getInertiaTensorWorld();
  }
	forces.resize(bodyNum);
	torques.resize(bodyNum);
	clearForceAndTorque();
	updateMassCenterPosition();

  n_dof_actuator = 3 * n_actuator_joint;

  m_grad_vn_to_v0.resize( bodyNum);
  m_grad_vn_to_omega0.resize( bodyNum);
  m_grad_omega_n_to_omega0.resize( bodyNum);
  m_grad_omega_n_to_v0.resize( bodyNum);

  m_grad_xn_to_v0.resize( bodyNum);
  m_grad_qn_to_v0.resize( bodyNum);
  m_grad_xn_to_omega0.resize( bodyNum);
  m_grad_qn_to_omega0.resize( bodyNum);

  for(int i = 0; i <  bodyNum; i++)
  {
    m_grad_vn_to_v0[i].resize(bodyNum);
    m_grad_omega_n_to_omega0[i].resize(bodyNum);
    m_grad_vn_to_omega0[i].resize(bodyNum);
    m_grad_omega_n_to_v0[i].resize(bodyNum);

    m_grad_xn_to_v0[i].resize( bodyNum);
    m_grad_qn_to_v0[i].resize( bodyNum);
    m_grad_xn_to_omega0[i].resize( bodyNum);
    m_grad_qn_to_omega0[i].resize( bodyNum);
  }

  reset();

  isChainRulePerformed = true;
}

void ArticulatedDynamicSystemBase::clearForceAndTorque()
{
	for (auto &&force : forces)
		force.setZero();
	for (auto &&torque : torques)
		torque.setZero();
}

void ArticulatedDynamicSystemBase::addForce(const Vector3r &f, RigidBodyObject *rbo)
{
	unsigned int rboIndex = this->getRigidIndex(rbo);
	forces[rboIndex] += f;
}
void ArticulatedDynamicSystemBase::addTorque(const Vector3r &t, RigidBodyObject *rbo)
{
	unsigned int rboIndex = this->getRigidIndex(rbo);
	torques[rboIndex] += t;
}

Matrix43r ArticulatedDynamicSystemBase::compute_grad_qn_to_v(unsigned parentRboIdx, unsigned childRboIdx, Real dt) 
{
  auto grad_quaternion_to_v0 = m_grad_qn_to_v0[parentRboIdx][childRboIdx];

  auto omega = bodies[parentRboIdx]->getAngularVelocity(); 
  auto q = bodies[parentRboIdx]->getRotation();
  auto p = Quaternionr(0., omega[0], omega[1], omega[2]); 
  Matrix4r grad_p_q_product_to_q = get_grad_p_q_product_to_q(p);
  Matrix43r grad_p_q_product_to_omega = get_grad_omega_q_product_to_omega(q); 

  Quaternionr new_q = q;
  new_q.coeffs() += dt * 0.5 * (p * new_q).coeffs();
  auto new_qn = new_q.normalized();
  auto new_qnv = Vector4r(new_qn.w(), new_qn.x(), new_qn.y(), new_qn.z());
  Matrix4r grad_qn_to_q =(Matrix4r::Identity() - new_qnv * new_qnv.transpose()) / new_q.norm();

  grad_quaternion_to_v0 = grad_qn_to_q * (grad_quaternion_to_v0 + dt / 2.
    * (grad_p_q_product_to_omega * m_grad_omega_n_to_v0[parentRboIdx][childRboIdx]   
     +  grad_p_q_product_to_q * grad_quaternion_to_v0)
  ) ;

  m_grad_qn_to_v0[parentRboIdx][childRboIdx] = grad_quaternion_to_v0;
  return m_grad_qn_to_v0[parentRboIdx][childRboIdx];
}

void ArticulatedDynamicSystemBase::timeStepLagrangian()
{
  LOG_INFO << YellowHead() << "========= use lagrangian in articluated system ====== " << YellowTail();
  const Real dt = SPH::TimeManager::getCurrent()->getTimeStepSize();
	size_t bodyNum = bodies.size();
	size_t jointNum = joints.size();
  size_t massMatrixSize = 6 * bodyNum;
	size_t constraintSize = 3 * jointNum;

  MatrixXr constraintMatrix(constraintSize, massMatrixSize);
  constraintMatrix.setZero();
  for(size_t i = 0; i < jointNum; i++)
  {
    auto &joint = joints[i];
    constraintMatrix.block(3 * i, 6 * joint.index1, 3, 3) = Matrix3r::Identity();
    constraintMatrix.block(3 * i, 6 * joint.index1 + 3, 3, 3) = - skewMatrix(bodies[joint.index1]->getWorldSpaceRotation() * joint.r_body1_to_joint); 
    constraintMatrix.block(3 * i, 6 * joint.index2, 3, 3) = -Matrix3r::Identity();
    constraintMatrix.block(3 * i, 6 * joint.index2 + 3, 3, 3) = skewMatrix(bodies[joint.index2]->getWorldSpaceRotation() * joint.r_body2_to_joint); 
  }

  for (size_t i = 0; i < bodyNum; i++)
  {
    auto &rbo = bodies[i];      
    massMatrix.block(6 * i, 6*i, 3, 3) = Matrix3r::Identity() * rbo->getMass();
		massMatrix.block(6 * i + 3, 6 * i + 3, 3, 3) = rbo->getInertiaTensorWorld();
  }

  VectorXr Vn(massMatrixSize); 
  Vn.setZero();
  VectorXr Fn(massMatrixSize); 
  Fn.setZero();
  for(size_t i = 0; i < bodyNum; i++)
  {
    auto &rbo = bodies[i];
    Vn.segment(6 * i, 3) = rbo->getVelocity() ; 
    Vn.segment(6*i+3, 3) = rbo->getAngularVelocity() ;

    Fn.segment(6 * i, 3) = forces[i]; 
    Fn.segment(6*i+3, 3) = torques[i] ;
  }

  MatrixXr J = constraintMatrix; 
  MatrixXr M = massMatrix;
  MatrixXr invM = M.inverse();
  MatrixXr Jr = J.transpose() * (J * invM * J.transpose()).inverse() * J; 

  MatrixXr invMJr = invM * Jr;
  MatrixXr Ir; 
  Ir.resizeLike(invMJr);
  Ir.setIdentity();
  
  VectorXr new_V = (Ir - invMJr)  * Vn + dt * (invM - invMJr * invM ) * Fn; 
  //LOG_INFO << Utilities::RedHead() << "m_grad_Vn_to_V0 before = \n" << m_grad_Vn_to_V0 << Utilities::YellowTail();

  for(size_t i = 0; i < bodyNum; i++)
  {
    auto &rbo = bodies[i];
    rbo->setVelocity(new_V.segment(6 * i, 3));
    rbo->setAngularVelocity(new_V.segment(6 *i + 3, 3));
  }
  clearForceAndTorque();
}

void ArticulatedDynamicSystemBase::timeStepImpulseBased()
{
  const Real dt = SPH::TimeManager::getCurrent()->getTimeStepSize();
	size_t bodyNum = bodies.size();
	size_t jointNum = joints.size();
  // -----------------------------------------
  // First, update rbo based on external forces (force from fluid, etc.)
  for(size_t i = 0; i < bodyNum; i++)
  {
    RigidBodyObject* &rbo = bodies[i];      
    rbo->setVelocity(rbo->getVelocity() + dt * 1.0 / rbo->getMass() * forces[i]);
    // for 2d, we don't need to consider gyroscopic forces 
    rbo->setAngularVelocity(rbo->getAngularVelocity() + dt * rbo->getInertiaTensorInverseWorld()* torques[i]);
  }
  clearForceAndTorque();

  // ---------------------------------------------------------------------------------------
  // Add the gradient of velocities from fluid-rigid coupling force
  for(int i = 0; i < joints.size(); i++)
  {
    auto joint = joints[i];
    unsigned parentIdx = joint.index1;
    unsigned childIdx = joint.index2;
    auto childRbo = bodies[childIdx];
    auto child_bm = rigidBodyInfo[childRbo].second; //boundary model of childRbo
    auto parentRbo = bodies[parentIdx]; 
    auto parent_bm = rigidBodyInfo[parentRbo].second; 

    //m_grad_vn_to_omega0[parentIdx][childIdx] += dt * 1.0 / childRbo->getMass() *
      //(parent_bm->get_grad_net_force_to_omega() * m_grad_omega_n_to_omega0[parentIdx][childIdx] + 
       //parent_bm->get_grad_net_force_to_v() * m_grad_vn_to_omega0[parentIdx][childIdx]);
    //m_grad_omega_n_to_omega0[parentIdx][childIdx] += dt * parentRbo->getInertiaTensorInverseWorld() *
        //(parent_bm->get_grad_net_torque_to_omega() * m_grad_omega_n_to_omega0[parentIdx][childIdx] + 
         //parent_bm->get_grad_net_torque_to_v() * m_grad_vn_to_omega0[parentIdx][childIdx]);

    //m_grad_vn_to_omega0[childIdx][childIdx] += dt * 1.0 / childRbo->getMass() * 
      //( child_bm->get_grad_net_force_to_omega() * m_grad_omega_n_to_omega0[childIdx][childIdx] + 
        //child_bm->get_grad_net_force_to_v() * m_grad_vn_to_omega0[childIdx][childIdx]); // FIXME: need to add all partial derivatives 
    //m_grad_omega_n_to_omega0[childIdx][childIdx] += dt * childRbo->getInertiaTensorInverseWorld() *
        //(child_bm->get_grad_net_torque_to_omega() * m_grad_omega_n_to_omega0[childIdx][childIdx] + 
         //child_bm->get_grad_net_torque_to_v() * m_grad_vn_to_omega0[childIdx][childIdx]);

    m_grad_vn_to_v0[parentIdx][childIdx] += dt * 1.0 / childRbo->getMass() *
      (parent_bm->get_grad_net_force_to_omega() * m_grad_omega_n_to_v0[parentIdx][childIdx] + 
       parent_bm->get_grad_net_force_to_v() * m_grad_vn_to_v0[parentIdx][childIdx]);
    m_grad_omega_n_to_v0[parentIdx][childIdx] += dt * parentRbo->getInertiaTensorInverseWorld() *
        (parent_bm->get_grad_net_torque_to_omega() * m_grad_omega_n_to_v0[parentIdx][childIdx] + 
         parent_bm->get_grad_net_torque_to_v() * m_grad_vn_to_v0[parentIdx][childIdx]);

    m_grad_vn_to_v0[childIdx][childIdx] += dt * 1.0 / childRbo->getMass() * 
      ( child_bm->get_grad_net_force_to_omega() * m_grad_omega_n_to_v0[childIdx][childIdx] + 
        child_bm->get_grad_net_force_to_v() * m_grad_vn_to_v0[childIdx][childIdx]); // FIXME: need to add all partial derivatives 
    m_grad_omega_n_to_v0[childIdx][childIdx] += dt * childRbo->getInertiaTensorInverseWorld() *
        (child_bm->get_grad_net_torque_to_omega() * m_grad_omega_n_to_v0[childIdx][childIdx] + 
         child_bm->get_grad_net_torque_to_v() * m_grad_vn_to_v0[childIdx][childIdx]);
  }
  // -----------------------------------------
  // Next, fulfill constraint by impulse-based method 
  std::vector<Vector3r> joint_impulse; 
  joint_impulse.reserve(joints.size());
  for(int i = 0; i < joints.size(); i++)
  {
    auto joint = joints[i];
    auto parentRbo = bodies[joint.index1];
    auto childRbo = bodies[joint.index2];

    Vector3r RL1 = parentRbo->getWorldSpaceRotation() * joint.r_body1_to_joint; 
    Vector3r RL2 = childRbo->getWorldSpaceRotation() * joint.r_body2_to_joint; 

    // new position of joint point on parent body and child body
    Vector3r parent_joint_new_x = parentRbo->getPosition() + dt * parentRbo->getVelocity() 
                + RL1 + dt * parentRbo->getAngularVelocity().cross(RL1);
    Vector3r child_joint_new_x = childRbo->getPosition() + dt * childRbo->getVelocity() 
                + RL2 + dt * childRbo->getAngularVelocity().cross(RL2);
    Vector3r delta_joint_x = parent_joint_new_x - child_joint_new_x;
    //LOG_INFO << Utilities::RedHead() <<  "delta_x = " << delta_x << Utilities::RedTail();

    Matrix3r K_parent = 1.0 / parentRbo->getMass() * Matrix3r::Identity() 
                    - skewMatrix(RL1) * parentRbo->getInertiaTensorInverseWorld() * skewMatrix(RL1);
    Matrix3r K_child = 1.0 / childRbo->getMass() * Matrix3r::Identity() 
                    - skewMatrix(RL2) * childRbo->getInertiaTensorInverseWorld() * skewMatrix(RL2);
    Matrix3r K = K_parent + K_child;
    Matrix3r invK = K.inverse();
    Vector3r impulse = 1.0 / dt * invK * delta_joint_x ;
    joint_impulse.push_back(impulse);

    Vector3r parent_new_v = parentRbo->getVelocity() + 1.0 / parentRbo->getMass() * - impulse;  
    Vector3r parent_new_omega = parentRbo->getAngularVelocity() + parentRbo->getInertiaTensorInverseWorld() * skewMatrix(RL1) * - impulse;  
    Vector3r child_new_v = childRbo->getVelocity() + 1.0 / childRbo->getMass() * impulse;  
    Vector3r child_new_omega = childRbo->getAngularVelocity() + childRbo->getInertiaTensorInverseWorld() * skewMatrix(RL2) * impulse;  
    // Note: cannot update velocities here since we need the old value in gradient computation

    //LOG_INFO << CyanHead() << "impulse = " << impulse.transpose() << YellowTail();
    //LOG_INFO << CyanHead() << "parent new v = " << parent_new_v.transpose() << YellowTail();
    //LOG_INFO << CyanHead() << "invK = \n" << invK << YellowTail();
    // ------------------------------------------------------------------------------------------------------------ 
    // derivatives
    // ------------------------------------------------------------------------------------------------------------ 
    unsigned parentRboIdx = joint.index1;
    unsigned childRboIdx = joint.index2;

    Matrix34r grad_RL1_to_parent_q = get_grad_Rqp_to_q(parentRbo->getRotation(), joint.r_body1_to_joint); 
    Matrix34r grad_RL2_to_child_q = get_grad_Rqp_to_q(childRbo->getRotation(), joint.r_body2_to_joint); 

    Matrix3r grad_RL1_to_child_v0 = grad_RL1_to_parent_q * m_grad_qn_to_v0[parentRboIdx][childRboIdx];
    Matrix3r grad_RL2_to_child_v0 = grad_RL2_to_child_q * m_grad_qn_to_v0[childRboIdx][childRboIdx];

    Matrix3r grad_delta_x_to_child_v0 = (m_grad_xn_to_v0[parentRboIdx][childRboIdx] + dt * m_grad_vn_to_v0[parentRboIdx][childRboIdx]
                              + grad_RL1_to_child_v0 
                              + dt * (skewMatrix(RL1).transpose() * m_grad_omega_n_to_v0[parentRboIdx][childRboIdx]
                              + skewMatrix(parentRbo->getAngularVelocity()) * grad_RL1_to_child_v0 ))

                            - (m_grad_xn_to_v0[childRboIdx][childRboIdx] + dt * m_grad_vn_to_v0[childRboIdx][childRboIdx]
                              + grad_RL2_to_child_v0 
                              + dt * (skewMatrix(RL2).transpose() * m_grad_omega_n_to_v0[childRboIdx][childRboIdx]
                              + skewMatrix(childRbo->getAngularVelocity()) * grad_RL2_to_child_v0) );

    //LOG_INFO << RedHead() << "grad_delta_x_to_child_omega0 = \n" << grad_delta_x_to_child_omega0<< YellowTail();

    // --------------------------------------------------------
    //Matrix3r grad_invK_delta_x_to_child_omega0 = Matrix3r::Zero(); // TODO
    Vector3r invK_delta_x = invK * delta_joint_x; 

    // Note: (grad invK)  * delta_x = -invK * (grad K) * (inv K * delta_x), so we need to differentiate K here  
    Matrix3r grad_K_invK_delta_x_to_child_v0 = - (skewMatrix(parentRbo->getInertiaTensorInverseWorld() * skewMatrix(RL1) * invK_delta_x).transpose() * grad_RL1_to_child_v0 
                            + (skewMatrix(RL1) * parentRbo->getInertiaTensorInverseWorld()) *skewMatrix(invK_delta_x).transpose() * grad_RL1_to_child_v0)
                                                    
                                                  - (skewMatrix(childRbo->getInertiaTensorInverseWorld() * skewMatrix(RL2) * invK_delta_x).transpose() * grad_RL2_to_child_v0 
                            + (skewMatrix(RL2) * childRbo->getInertiaTensorInverseWorld()) *skewMatrix(invK_delta_x).transpose() * grad_RL2_to_child_v0);

    Matrix3r grad_invK_delta_x_to_child_v0 = - invK * grad_K_invK_delta_x_to_child_v0;
                        
    // --------------------------------------------------------
    Matrix3r grad_impulse_to_child_v0 = 1.0 / dt * (invK * grad_delta_x_to_child_v0 + grad_invK_delta_x_to_child_v0);

    // Only compute the gradient flow from childRbo --> parentRbo
    m_grad_vn_to_v0[parentRboIdx][childRboIdx] +=  1.0 / parentRbo->getMass() * - grad_impulse_to_child_v0;
    m_grad_vn_to_v0[childRboIdx][childRboIdx] +=  1.0 / childRbo->getMass() * grad_impulse_to_child_v0;

    m_grad_omega_n_to_v0[parentRboIdx][childRboIdx] += parentRbo->getInertiaTensorInverseWorld() *
                  (skewMatrix(RL1) * - grad_impulse_to_child_v0 +  
                   skewMatrix( - impulse).transpose() *  grad_RL1_to_child_v0 );
    m_grad_omega_n_to_v0[childRboIdx][childRboIdx] += childRbo->getInertiaTensorInverseWorld() *
                  (skewMatrix(RL2) * grad_impulse_to_child_v0 +  
                  skewMatrix(impulse).transpose() * grad_RL2_to_child_v0);

    //LOG_INFO << YellowHead() << "m_grad_vn_to_v0[parentRboIdx][childRboIdx] = \n" << m_grad_vn_to_v0[parentRboIdx][childRboIdx] << YellowTail();
    //LOG_INFO << GreenHead() << "m_grad_omega_n_to_v0[parentRboIdx][childRboIdx] = \n" << m_grad_omega_n_to_v0[parentRboIdx][childRboIdx] << YellowTail();
      
    // --------------------------------------------------------
    // Finally, update velocities
    parentRbo->setVelocity(parent_new_v);
    parentRbo->setAngularVelocity(parent_new_omega);
    childRbo->setVelocity(child_new_v);
    childRbo->setAngularVelocity(child_new_omega);
  }


  // ---------------------------------------------------------------------------------------
  //
  for(int i = 0; i < joints.size(); i++)
  {
    auto joint = joints[i];
    unsigned parentRboIdx = joint.index1;
    unsigned childRboIdx = joint.index2;

    m_grad_xn_to_v0[parentRboIdx][childRboIdx] += dt * m_grad_vn_to_v0[parentRboIdx][childRboIdx];
    m_grad_xn_to_v0[childRboIdx][childRboIdx] += dt * m_grad_vn_to_v0[childRboIdx][childRboIdx];

    compute_grad_qn_to_v(parentRboIdx, childRboIdx, dt);
    compute_grad_qn_to_v(childRboIdx, childRboIdx, dt);

    //LOG_INFO << YellowHead() << "m_grad_xn_to_v0[parentRboIdx][childRboIdx] = \n" << m_grad_xn_to_v0[parentRboIdx][childRboIdx] << YellowTail();
    //LOG_INFO << GreenHead() << "m_grad_qn_to_v0[parentRboIdx][childRboIdx] = \n" << m_grad_qn_to_v0[parentRboIdx][childRboIdx] << YellowTail() << "\n";
    Simulation *sim = Simulation::getCurrent();
    if (sim->isDebug())
    {
      LOG_INFO << RedHead() << "m_grad_xn_to_v0[" << parentRboIdx << "][" <<  childRboIdx << "] = \n" << m_grad_xn_to_v0[parentRboIdx][childRboIdx] << "\n" << YellowTail();
      LOG_INFO << GreenHead() << "m_grad_xn_to_v0[" << childRboIdx << "][" <<  childRboIdx << "] = \n" << m_grad_xn_to_v0[childRboIdx][childRboIdx] << "\n" << YellowTail();

      LOG_INFO << CyanHead() << "m_grad_qn_to_v0[" << parentRboIdx << "][" <<  childRboIdx << "] = \n" << m_grad_qn_to_v0[parentRboIdx][childRboIdx] << "\n" << YellowTail();
      LOG_INFO << "m_grad_qn_to_v0[" << childRboIdx << "][" <<  childRboIdx << "] = \n" << m_grad_qn_to_v0[childRboIdx][childRboIdx];
      LOG_INFO << " ==================================" ; 
        
    }
  }

  isChainRulePerformed = true;

}

void ArticulatedDynamicSystemBase::timeStep()
{
  if(m_useLagrangian)
  {
    timeStepLagrangian();
  }
  else {
    timeStepImpulseBased();
  }
   
}

void ArticulatedDynamicSystemBase::updateMassCenterPosition()
{
  massCenterPosition.setZero();
  Real totalMass = 0;
	for (auto &rbo : bodies)
	{
		massCenterPosition += rbo->getMass() * rbo->getPosition();
		totalMass += rbo->getMass();
	}
	massCenterPosition /= totalMass;
}

void ArticulatedDynamicSystemBase::reset()
{
  //TODO 
  for(int i = 0; i < bodies.size(); i++)
  {
    for(int j = 0; j < bodies.size(); j++)
    {
      m_grad_vn_to_v0[i][j].setZero();
      m_grad_vn_to_omega0[i][j].setZero();
      m_grad_omega_n_to_omega0[i][j].setZero();
      m_grad_omega_n_to_v0[i][j].setZero();

      m_grad_xn_to_v0[i][j].setZero();
      m_grad_qn_to_v0[i][j].setZero();
      m_grad_xn_to_omega0[i][j].setZero();
      m_grad_qn_to_omega0[i][j].setZero();
    }
  }

  for(int i = 0; i < bodies.size(); i++)
  {
    m_grad_vn_to_v0[i][i].setIdentity();
    m_grad_omega_n_to_omega0[i][i].setIdentity();
  }
 
  clearForceAndTorque();
  updateMassCenterPosition();
}


void ArticulatedDynamicSystemBase::checkJointsStatus()
{
	std::cout << "--------------\nCheck joints:\n";
	for (auto &joint : joints)
	{
		auto bd1 = bodies[joint.index1], bd2 = bodies[joint.index2];
		Vector3r relativeVelocity = (bd1->getVelocity() + bd1->getAngularVelocity().cross(bd1->getWorldSpaceRotation() * joint.r_body1_to_joint)) -
									(bd2->getVelocity() + bd2->getAngularVelocity().cross(bd2->getWorldSpaceRotation() * joint.r_body2_to_joint));
		Vector3r relativePosition = (bd1->getPosition() + bd1->getWorldSpaceRotation() * joint.r_body1_to_joint) -
									(bd2->getPosition() + bd2->getWorldSpaceRotation() * joint.r_body2_to_joint);
		std::cout << "Velocity:\t" << relativeVelocity[0] << '\t' << relativeVelocity[1] << '\t' << relativeVelocity[2] << '\n';
		std::cout << "Position:\t" << relativePosition[0] << '\t' << relativePosition[1] << '\t' << relativePosition[2] << '\n';
	}
	std::cout << "--------------\n";
}

void ArticulatedDynamicSystemBase::getJointInfoByBody(RigidBodyObject *body, RigidBodyObject *&otherBody, Vector3r &l1, Vector3r &l2)
{
    auto index = getRigidIndex(body);
    int count = 0;
    Balljoint *chosenJoint = nullptr;
    // Summarized the related joints to body
    for(auto &joint : joints)
    {
      if(joint.index1 == index || joint.index2 == index)
      {
        count += 1;
        chosenJoint = &joint;
      }
    }

    if(count >= 2) {
      std::cout << "Warning: The rigid body is connected to more than one joints, you cannot set angular velocity relative to the joint\n";
      return;
    }
    else if(!chosenJoint){
      std::cout << "Warning: No joints for rigid body";
      return;
    }
    else{
      if(chosenJoint->index1 == index)
      {
        l1 = chosenJoint->r_body1_to_joint;
        l2 = chosenJoint->r_body2_to_joint;
        otherBody = bodies[chosenJoint->index2];
      }
      else {
        l1 = chosenJoint->r_body2_to_joint;
        l2 = chosenJoint->r_body1_to_joint;
        otherBody = bodies[chosenJoint->index1];
      }
    }
}

void ArticulatedDynamicSystemBase::setAngularVelocityToJoint(RigidBodyObject *body, const Vector3r &omega)
{
  RigidBodyObject *otherBody = nullptr;
  Vector3r L1, L2, RL1, RL2;
  getJointInfoByBody(body, otherBody, L1, L2);
  RL1 = body->getRotation().toRotationMatrix() * L1;
  RL2 = body->getRotation().toRotationMatrix() * L2;
  
  body->setAngularVelocity(omega); 
  // The constraint: v1 + omega1 cross L1 = v2 + omega2 cross L2
  //body->setVelocity(otherBody->getVelocity() + otherBody->getAngularVelocity().cross(RL2) - omega.cross(RL1));
  //Vector3r new_v = -omega.cross(RL1); 
  //body->setVelocity(new_v);
  //otherBody->setVelocity(- new_v * body->getMass() / otherBody->getMass() + otherBody->getVelocity());
}

void ArticulatedDynamicSystemBase::addAngularVelocityToJoint(RigidBodyObject *body, const Vector3r &delta_omega)
{
  RigidBodyObject *otherBody = nullptr;
  Vector3r L1, L2, RL1, RL2;
  getJointInfoByBody(body, otherBody, L1, L2);
  RL1 = body->getRotation().toRotationMatrix() * L1;
  RL2 = body->getRotation().toRotationMatrix() * L2;
  
  body->setAngularVelocity(delta_omega + body->getAngularVelocity());
  // The constraint: v1 + omega1 cross L1 = v2 + omega2 cross L2
  //body->setVelocity(otherBody->getVelocity() + otherBody->getAngularVelocity().cross(RL2) - body->getAngularVelocity().cross(RL1));
  //Vector3r new_v = -delta_omega.cross(RL1) + body->getVelocity();
  //body->setVelocity(new_v);
  //otherBody->setVelocity(- new_v * body->getMass() / otherBody->getMass() + otherBody->getVelocity());
}

bool ArticulatedDynamicSystemBase::findRigidIndex(RigidBodyObject *rbo, unsigned int &index)
{
	assert(rbo->getArticulatedSystem() == this);
  auto find = rigidBodyInfo.find(rbo);
  if(find == rigidBodyInfo.end())
    return false;
  index = find->second.first;
  return true;
}

unsigned int ArticulatedDynamicSystemBase::getRigidIndex(RigidBodyObject *rbo)
{
  unsigned int index = 0;
  if (!findRigidIndex(rbo, index))
  {
    std::cout << "Somthing wrong with articulated rigid body system\n";
    std::abort();
  }
  return index;
}

bool ArticulatedDynamicSystemBase::findBoundaryModel(RigidBodyObject *rbo, BoundaryModel_Akinci2012 *&bm)
{
	assert(rbo->getArticulatedSystem() == this);
  auto find = rigidBodyInfo.find(rbo);
  if(find == rigidBodyInfo.end())
    return false;
  bm = find->second.second;
  return true;
}

BoundaryModel_Akinci2012 *ArticulatedDynamicSystemBase::getBoundaryModel(RigidBodyObject *rbo)
{
  BoundaryModel_Akinci2012* bm = nullptr;
  if (!findBoundaryModel(rbo, bm))
  {
    std::cout << "Somthing wrong with articulated rigid body system\n";
    std::abort();
  }
  return bm;
}

std::vector<Vector3r> ArticulatedDynamicSystemBase::getJointPoints()
{
	std::vector<Vector3r> results;
	results.reserve(joints.size());
	for (auto &joint : joints)
	{
		auto &bd2 = bodies[joint.index2];
		Vector3r joint_position = (bd2->getPosition() + bd2->getWorldSpaceRotation() * joint.r_body2_to_joint);
		results.push_back(joint_position);
	}
	return results;
}

void ArticulatedDynamicSystemBase::perform_chain_rule(const Real dt)
{
  if(!isChainRulePerformed)
  {
    //MatrixXr grad_F_to_V0;
    //grad_F_to_V0.resizeLike(m_grad_Vn_to_V0);
    //MatrixXr grad_F_to_actuator_omega0;
    //grad_F_to_actuator_omega0.resizeLike(m_grad_Vn_to_actuator_omega0); // n_dof_actuator x 6 * bodyNum
    //grad_F_to_actuator_omega0.setZero();

    //for (unsigned int i = 0; i < bodies.size(); i++)
    //{
      //auto body = bodies[i];
      //auto bm = getBoundaryModel(body);
      //// here approximately calculate dfn/dv0 as (dfn/dvn)(dvn/dv0)
      //// dfn/dvn ~ \partial fn/\partial vn, \partial fn/\partial xn
      //Matrix6r grad_Fn_to_Vn;
      //grad_Fn_to_Vn.block(0, 0, 3, 3) = bm->get_total_grad_force_to_v();
      //grad_Fn_to_Vn.block(3, 0, 3, 3) = bm->get_total_grad_force_to_omega();
      //grad_Fn_to_Vn.block(0, 3, 3, 3) = bm->get_total_grad_torque_to_v();
      //grad_Fn_to_Vn.block(3, 3, 3, 3) = bm->get_total_grad_torque_to_omega();
      ////if(isArtSystemDebug)
      ////{
        ////std::cout << i << "\tgrad_f_to_v0" << '\n'
                  ////<< grad_Fn_to_Vn << '\n';
      ////}
      ////grad_F_to_V0.middleRows(6 * i, 6) = grad_Fn_to_Vn * m_grad_Vn_to_V0.middleRows(6 * i, 6);
      //grad_F_to_actuator_omega0.middleCols(6 * i, 6) = grad_Fn_to_Vn * m_grad_Vn_to_actuator_omega0.middleCols(6 * i, 6);
    //}
    //LOG_INFO << Utilities::RedHead() << "m_grad_Vn_to_V0 before = \n" << m_grad_Vn_to_V0 << Utilities::YellowTail();
    //m_grad_Vn_to_V0 = m_grad_Vn_to_V0 * m_grad_Vn_to_Vn_1;

    //LOG_INFO << Utilities::CyanHead() << "m_grad_Vn_to_Vn_1= " << m_grad_Vn_to_Vn_1.norm()<< Utilities::CyanTail();
    //LOG_INFO << Utilities::CyanHead() << "m_grad_Vn_to_Fn_1= " << m_grad_Vn_to_Fn_1.norm()<< Utilities::CyanTail();

    //m_grad_Vn_to_actuator_omega0 = m_grad_Vn_to_actuator_omega0 * m_grad_Vn_to_Vn_1
            //+ grad_F_to_actuator_omega0 * m_grad_Vn_to_Fn_1;


    //m_grad_rootRB_vn_to_actuator_omega0 = m_grad_Vn_to_actuator_omega0.middleCols(0, 3);
    // only for test
    //m_grad_rootRB_vn_to_actuator_omega0 = m_grad_Vn_to_V0.block(9, 0, 3, 3);
    //m_grad_rootRB_xn_to_actuator_omega0 = m_grad_rootRB_xn_to_actuator_omega0 + dt * m_grad_rootRB_vn_to_actuator_omega0;


    //LOG_INFO << Utilities::GreenHead() << "m_grad_Vn_to_Vn_1 = \n" << m_grad_Vn_to_Vn_1 << Utilities::YellowTail();
    ////LOG_INFO << Utilities::RedHead() << "mgrad_Vn_to_actuator_omega0 = \n" << m_grad_Vn_to_actuator_omega0 << Utilities::YellowTail();
    //LOG_INFO << Utilities::RedHead() << "m_grad_Vn_to_V0 = \n" << m_grad_Vn_to_V0 << Utilities::YellowTail();
    //LOG_INFO << Utilities::CyanHead() << "m_grad_rootRB_vn_to_actuator_omega0 = \n" << m_grad_rootRB_vn_to_actuator_omega0 << Utilities::YellowTail();
    //LOG_INFO << Utilities::YellowHead() << "m_grad_rootRB_xn_to_actuator_omega0 = \n" << m_grad_rootRB_xn_to_actuator_omega0 << Utilities::YellowTail();



    //for (unsigned int i = 0; i < bodies.size(); i++)
    //{
      //m_grad_Xn_to_V0.middleRows(7 * i, 3) = m_grad_Xn_to_V0.middleRows(7 * i, 3) + dt * m_grad_Vn_to_V0.middleRows(6 * i, 3);

      //auto body = bodies[i];
      //auto bm = getBoundaryModel(body);
      //auto q = body->getRotation();
      //auto omega = body->getAngularVelocity();
      //auto p = Quaternionr(0., omega[0], omega[1], omega[2]); 
      //Quaternionr new_q = q;
      //new_q.coeffs() += dt * 0.5 * (p * new_q).coeffs();
      //auto new_qn = new_q.normalized();
      //auto new_qnv = Vector4r(new_qn.w(), new_qn.x(), new_qn.y(), new_qn.z());

      //Matrix4r grad_p_q_product_to_q = (Matrix4r() << p.w(), -p.x(), -p.y(), -p.z(),
                                        //p.x(), p.w(), -p.z(), p.y(),
                                        //p.y(), p.z(), p.w(), -p.x(),
                                        //p.z(), -p.y(), p.x(), p.w())
                                           //.finished();
      //Matrix43r grad_p_q_product_to_omega = (Matrix43r() << -q.x(), -q.y(), -q.z(),
                                             //q.w(), q.z(), -q.y(),
                                             //-q.z(), q.w(), q.x(),
                                             //q.y(), -q.x(), q.w())
                                                //.finished();

      //Matrix4r grad_qn_to_q = (Matrix4r::Identity() - new_qnv * new_qnv.transpose()) / new_q.norm();

      //MatrixXr grad_xn_to_v0_current_quaternion_part = m_grad_xn_to_v0.middleRows(7 * i + 3, 4);
      //MatrixXr grad_vn_to_v0_current_omega_part = m_grad_vn_to_v0.middleRows(6 * i + 3, 3);

      //if (false && isArtSystemDebug)
      //{
        //std::cout << "grad_p_q_product_to_q\n"
                  //<< grad_p_q_product_to_q << "\n\ngrad_p_q_product_to_omega\n"
                  //<< grad_p_q_product_to_omega << "\n\ngrad_qn_to_q\n"
                  //<< grad_qn_to_q << "\n";

        //std::cout << "grad_xn_to_v0_current_quaternion_part:\n"
                  //<< grad_xn_to_v0_current_quaternion_part << "\n\ngrad_vn_to_v0_current_omega_part\n"
                  //<< grad_vn_to_v0_current_omega_part << "\n\n";
      //}

      //MatrixXr tempResult = dt / 2. * (grad_p_q_product_to_q * grad_xn_to_v0_current_quaternion_part + grad_p_q_product_to_omega * grad_vn_to_v0_current_omega_part);

      //m_grad_xn_to_v0.middleRows(7 * i + 3, 4) = grad_qn_to_q * (grad_xn_to_v0_current_quaternion_part + tempResult);
    //}
    isChainRulePerformed = true;
    //if (isArtSystemDebug)
    //{
      //std::cout << "after chain rule:\n";
      //std::cout << "m_grad_vn_to_v0\n"
                //<< m_grad_vn_to_v0 << "\n\n";
    // std::cout << "m_grad_xn_to_v0\n"
    //<< m_grad_xn_to_v0 << "\n\n";
    //}

    //LOG_INFO << Utilities::CyanHead() << "m_grad_Vn_to_actuator_omega0 = " << m_grad_Vn_to_actuator_omega0 << Utilities::CyanTail();
    //LOG_INFO << Utilities::CyanHead() << "m_grad_rootRB_vn_to_actuator_omega0 = " << m_grad_rootRB_vn_to_actuator_omega0 << Utilities::CyanTail();
    //LOG_INFO << Utilities::CyanHead() << "grad_rootRB_xn_to_actuator_omega0 = " << m_grad_rootRB_xn_to_actuator_omega0 << Utilities::CyanTail();
    
  }
}


//MatrixXr ArticulatedDynamicSystemBase::get_grad_rootRB_xn_to_actuator_omega0()
//{
  //return m_grad_rootRB_xn_to_actuator_omega0; 
//}

//Matrix3r ArticulatedDynamicSystemBase::get_grad_v_to_v0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2)
//{
  //unsigned int index1 = getRigidIndex(bm1->getRigidBodyObject());
  //unsigned int index2;
  //if (!findRigidIndex(bm2->getRigidBodyObject(), index2))
  //{
    //return Matrix3r::Zero();
  //}
  //return m_grad_vn_to_v0[index1][index2];
//}
//Matrix3r ArticulatedDynamicSystemBase::get_grad_omega_to_v0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2)
//{
  //unsigned int index1 = getRigidIndex(bm1->getRigidBodyObject());
  //unsigned int index2;
  //if (!findRigidIndex(bm2->getRigidBodyObject(), index2))
  //{
    //return Matrix3r::Zero();
  //}
  //Matrix3r result(m_grad_Vn_to_V0.block(6 * index1 + 3, 6 * index2, 3, 3));
  //if(isArtSystemDebug)
  //{
    //std::cout << index1 << '\t' << index2 << "\tdw/dv0\n"
              //<< result << '\n';
  //}
  //return result;
//}
//Matrix3r ArticulatedDynamicSystemBase::get_grad_x_to_v0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2)
//{
  //unsigned int index1 = getRigidIndex(bm1->getRigidBodyObject());
  //unsigned int index2;
  //if (!findRigidIndex(bm2->getRigidBodyObject(), index2))
  //{
    //return Matrix3r::Zero();
  //}
  //Matrix3r result(m_grad_Xn_to_V0.block(7 * index1, 6 * index2, 3, 3));
  //if(isArtSystemDebug)
  //{
    //std::cout << index1 << '\t' << index2 << "\tdx/dv0\n"
              //<< result << '\n';
  //}
  //return result;
//}
//Matrix43r ArticulatedDynamicSystemBase::get_grad_quaternion_to_v0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2)
//{
  //unsigned int index1 = getRigidIndex(bm1->getRigidBodyObject());
  //unsigned int index2;
  //if (!findRigidIndex(bm2->getRigidBodyObject(), index2))
  //{
    //return Matrix43r::Zero();
  //}
  //Matrix43r result(m_grad_Xn_to_V0.block(7 * index1 + 3, 6 * index2, 4, 3));
  //if(isArtSystemDebug)
  //{
    //std::cout << index1 << '\t' << index2 << "\tdq/dv0\n"
              //<< result << '\n';
  //}
  //return result;
//}

//Matrix3r ArticulatedDynamicSystemBase::get_grad_v_to_omega0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2)
//{
  //unsigned int index1 = getRigidIndex(bm1->getRigidBodyObject());
  //unsigned int index2;
  //if (!findRigidIndex(bm2->getRigidBodyObject(), index2))
  //{
    //return Matrix3r::Zero();
  //}
  //Matrix3r result(m_grad_Vn_to_V0.block(6 * index1, 6 * index2 + 3, 3, 3));
  //if(isArtSystemDebug)
  //{
    //std::cout << index1 << '\t' << index2 << "\tdv/dw0\n"
              //<< result << '\n';
  //}
  //return result;
//}
//Matrix3r ArticulatedDynamicSystemBase::get_grad_omega_to_omega0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2)
//{
  //unsigned int index1 = getRigidIndex(bm1->getRigidBodyObject());
  //unsigned int index2;
  //if (!findRigidIndex(bm2->getRigidBodyObject(), index2))
  //{
    //return Matrix3r::Zero();
  //}
  //Matrix3r result(m_grad_Vn_to_V0.block(6 * index1 + 3, 6 * index2 + 3, 3, 3));
  //if(isArtSystemDebug)
  //{
    //std::cout << index1 << '\t' << index2 << "\tdw/dw0\n"
              //<< result << '\n';
  //}
  //return result;
//}
Matrix3r ArticulatedDynamicSystemBase::get_grad_x_to_omega0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2)
{
  unsigned int index1 = getRigidIndex(bm1->getRigidBodyObject());
  unsigned int index2;
  if (!findRigidIndex(bm2->getRigidBodyObject(), index2))
  {
    return Matrix3r::Zero();
  }
  return m_grad_xn_to_omega0[index1][index2];
}
Matrix3r ArticulatedDynamicSystemBase::get_grad_x_to_v0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2)
{
  unsigned int index1 = getRigidIndex(bm1->getRigidBodyObject());
  unsigned int index2;
  if (!findRigidIndex(bm2->getRigidBodyObject(), index2))
  {
    return Matrix3r::Zero();
  }
  return m_grad_xn_to_v0[index1][index2];
}

Matrix43r ArticulatedDynamicSystemBase::get_grad_quaternion_to_v0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2)
{
  unsigned int index1 = getRigidIndex(bm1->getRigidBodyObject());
  unsigned int index2;
  if (!findRigidIndex(bm2->getRigidBodyObject(), index2))
  {
    return Matrix43r::Zero();
  }
  return m_grad_qn_to_v0[index1][index2];
}
//Matrix43r ArticulatedDynamicSystemBase::get_grad_quaternion_to_omega0(BoundaryModel_Akinci2012 *bm1, BoundaryModel_Akinci2012 *bm2)
//{
  //unsigned int index1 = getRigidIndex(bm1->getRigidBodyObject());
  //unsigned int index2;
  //if (!findRigidIndex(bm2->getRigidBodyObject(), index2))
  //{
    //return Matrix43r::Zero();
  //}
  //Matrix43r result(m_grad_Xn_to_V0.block(7 * index1 + 3, 6 * index2 + 3, 4, 3));
  //if(isArtSystemDebug)
  //{
    //std::cout << index1 << '\t' << index2 << "\tdq/dw0\n"
              //<< result << '\n';
  //}
  //return result;
//}

//void ArticulatedDynamicSystemBase::get_bm_grad_information(BoundaryModel_Akinci2012 *bm, Matrix3r &grad_v_to_v0, Matrix3r &grad_v_to_omega0, Matrix3r &grad_omega_to_v0, Matrix3r &grad_omega_to_omega0,
                                                           //Matrix3r &grad_x_to_v0, Matrix3r &grad_x_to_omega0, Matrix34r &grad_quaternion_to_v0, Matrix34r &grad_quaternion_to_omega0)
//{
  ////unsigned int index = getRigidIndex(bm->getRigidBodyObject());
  ////grad_v_to_v0 = m_grad_Vn_to_V0.block(6 * index, 6 * index, 3, 3).transpose();
  ////grad_v_to_omega0 = m_grad_Vn_to_V0.block(6 * index, 6 * index + 3, 3, 3).transpose();
  ////grad_omega_to_v0 = m_grad_Vn_to_V0.block(6 * index + 3, 6 * index, 3, 3).transpose();
  ////grad_omega_to_omega0 = m_grad_Vn_to_V0.block(6 * index + 3, 6 * index + 3, 3, 3).transpose();
  ////grad_x_to_v0 = m_grad_Xn_to_V0.block(7 * index, 6 * index, 3, 3).transpose();
  ////grad_x_to_omega0 = m_grad_Xn_to_V0.block(7 * index, 6 * index + 3, 3, 3).transpose();
  ////grad_quaternion_to_v0 = m_grad_Xn_to_V0.block(7 * index + 3, 6 * index, 4, 3).transpose();
  ////grad_quaternion_to_omega0 = m_grad_Xn_to_V0.block(7 * index + 3, 6 * index + 3, 4, 3).transpose();
//}
