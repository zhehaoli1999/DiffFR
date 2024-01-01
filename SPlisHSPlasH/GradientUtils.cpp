#include "GradientUtils.h"
#include "SPlisHSPlasH/Common.h"

Matrix3r skewMatrix(const Vector3r& v){
  return (Matrix3r() << 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0).finished();
}
// gradient of R(q)p to q, q is quaternion, p \in \R^3 
// Ref: Section3. Quaternions of "Position and Orientation Based Cosserat Rods"
Matrix34r get_grad_Rqp_to_q(Quaternionr q, Vector3r p){ 
  Vector3r qv = Vector3r(q.x(), q.y(), q.z());
  Vector3r tmp1 = 2. * (q.w() * p - p.cross(qv));
  Matrix3r tmp2 = 2. * (qv.dot(p) * Matrix3r::Identity() 
        + qv*p.transpose() - p*qv.transpose() - q.w() * skewMatrix(p));

  Matrix34r result = Matrix34r::Zero();
  result.middleCols(0, 1) = tmp1; 
  result.middleCols(1, 3) = tmp2; 
  return result;
 }

// gradient of R(q)^Tp to q 
Matrix34r get_grad_RqTp_to_q(Quaternionr q, Vector3r p)
{
  auto qv = Vector3r(q.x(), q.y(), q.z());
  Vector3r tmp1 = 2. * (q.w() * p + p.cross(qv));
  Matrix3r tmp2 = 2. * (qv.dot(p) * Matrix3r::Identity() 
        + qv*p.transpose() - p*qv.transpose() + q.w() * skewMatrix(p));
  
  Matrix34r result = Matrix34r::Zero();
  result.middleCols(0, 1) = tmp1; 
  result.middleCols(1, 3) = tmp2; 
  return result;

}

Matrix4r get_grad_p_q_product_to_q(Quaternionr p)
{
  return (Matrix4r() << 
        p.w(), -p.x(), -p.y(), -p.z(), 
        p.x(),  p.w(), -p.z(),  p.y(), 
        p.y(),  p.z(),  p.w(), -p.x(), 
        p.z(), -p.y(),  p.x(),  p.w()).finished();
}

Matrix43r get_grad_omega_q_product_to_omega(Quaternionr q)
{
  return (Matrix43r() << 
        -q.x(), -q.y(), -q.z(),
        q.w(), q.z(), -q.y(),
        -q.z(), q.w(), q.x(),
        q.y(), -q.x(), q.w()
    ).finished() ;
}
