#pragma once
#include "Common.h"

Matrix3r skewMatrix(const Vector3r& v);

// gradient of R(q)p to q, q is quaternion, p \in \R^3 
// Ref: Section3. Quaternions of "Position and Orientation Based Cosserat Rods"
Matrix34r get_grad_Rqp_to_q(Quaternionr q, Vector3r p);

Matrix34r get_grad_RqTp_to_q(Quaternionr q, Vector3r p);

Matrix4r get_grad_p_q_product_to_q(Quaternionr p);
Matrix43r get_grad_omega_q_product_to_omega(Quaternionr q);
