#pragma once
#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/TimeStep.h"
#include "SPlisHSPlasH/SPHKernels.h"
#include "SimulationDataDiffDFSPH.h"
#include "SPlisHSPlasH/BoundaryModel_Akinci2012.h"
#include "include/LBFGS.h"
#include <functional>

//#define OPTIM_ENABLE_EIGEN_WRAPPERS
//#include "optim.hpp"

#define BACKWARD

namespace SPH
{
class SimulationDataDiffDFSPH;

class TimeStepDiffDFSPH: public TimeStep
  {
  protected:
    SimulationDataDiffDFSPH m_simulationData;

    unsigned int m_counter;
    unsigned int step_count;
    const Real m_eps = static_cast<Real>(1.0e-5);
    bool m_enableDivergenceSolver;
    bool m_is_trajectory_finished_callback;
    bool m_is_in_new_trajectory;
    bool m_is_debug;
    bool m_optimize_rotation;
    bool m_use_MLS_pressure;
    bool m_use_symmetric_formula;
    bool m_use_strong_coupling;
    bool m_use_pressure_warmstart; 
    bool m_use_divergence_warmstart; 
    bool m_enable_count_neighbor_dof;
    unsigned int m_iterationsV;
    Real m_maxErrorV;
    unsigned int m_maxIterationsV;

    void computeDFSPHFactor(const unsigned int fluidModelIndex);
    void pressureSolve();
    int pressureSolveIteration(const unsigned int fluidModelIndex, Real &avg_density_err);
    void divergenceSolve();
    void divergenceSolveIteration(const unsigned int fluidModelIndex, Real &avg_density_err);
    void computeDensityAdv(const unsigned int fluidModelIndex, const unsigned int index, const int numParticles, const Real h, const Real density0);
    void computeDensityChange(const unsigned int fluidModelIndex, const unsigned int index, const Real h);

		void warmstartDivergenceSolve(const unsigned int fluidModelIndex);
		void warmstartPressureSolve(const unsigned int fluidModelIndex);
    /** Perform the neighborhood search for all fluid particles.
    */
    void performNeighborhoodSearch();
    virtual void emittedParticles(FluidModel *model, const unsigned int startIndex);

    virtual void initParameters();

#ifdef BACKWARD
    void computeGradient( BoundaryModel_Akinci2012* bm_neighbor,
                         const unsigned int fluidModelIndex,
                         const unsigned int pid,
                         const unsigned int fluidParticleIndex,
                         const unsigned int neighborIndex,
                         const Real b_i,
                         const Real ki,
                         const Vector3r force,
                         const unsigned int mode); // mode 0 for pressureSolve, 1
    // for divergenceSolve
    void computeRigidBodyGradient(const unsigned int fluidModelIndex, 
                                  const unsigned int mode);

    Vector3r computeGradFactorToBoundaryParticleX(const unsigned int fluidModelIndex,
                                                  const unsigned int pointSetId,
                                                  const unsigned int fluidParticleIndex, const unsigned int BoundaryParticleIndex);

    Vector3r computeGradDensityAdvToBoundaryParticleX(
      const unsigned int fluidModelIndex,
      const unsigned int pointSetId,
      const unsigned int fluidParticleIndex,
      const unsigned int BoundaryParticleIndex);
    Vector3r computeGradDensityChangeToBoundaryParticleX(const unsigned int fluidModelIndex,
                                                         const unsigned int pointSetId,
                                                         const unsigned int fluidParticleIndex, const unsigned int BoundaryParticleIndex);

    Vector3r computeGradDensityAdvToBoundaryParticleV(const unsigned int fluidModelIndex,
                                                      const unsigned int pointSetId,
                                                      const unsigned int fluidParticleIndex, const unsigned int BoundaryParticleIndex);
    Vector3r computeGradDensityChangeToBoundaryParticleV(const unsigned int fluidModelIndex,
                                                         const unsigned int pointSetId,
                                                         const unsigned int fluidParticleIndex, const unsigned int BoundaryParticleIndex);

    Vector3r computeGradDensityAdvToFluidParticleV(const unsigned int fluidModelIndex, 
                                                   const unsigned int fluidParticleIndex);

    Vector3r computeGradDensityChangeToFluidParticleV(const unsigned int fluidModelIndex, 
                                                   const unsigned int fluidParticleIndex);
    struct LBFGSFuncWrapper
    {
      LBFGSFuncWrapper(): lbfgs_solver(lbfgs_param) {}

      using FuncType = std::function<VectorXr(const VectorXr &)>;
      void setFunction(FuncType const& func)
      {
        m_func = func;
      }
      FuncType m_func;
      Real operator()(const VectorXr &x, VectorXr &grad);
      LBFGSpp::LBFGSParam<Real> lbfgs_param;
      LBFGSpp::LBFGSSolver<Real, LBFGSpp::LineSearchMoreThuente> lbfgs_solver;
      VectorXr initPoint;
      bool useNormalizedGrad = false;
    };

    LBFGSFuncWrapper lbfgs_wrapper;

#endif

  public:
    static int SOLVER_ITERATIONS_V;
    static int MAX_ITERATIONS_V;
    static int MAX_ERROR_V;
    static int USE_DIVERGENCE_SOLVER;
    static int USE_SYMMETRIC_FORMULA;
    static int USE_MLS_PRESSURE;
    static int USE_STRONG_COUPLING;
    static int USE_PRESSURE_WARMSTART;
    static int USE_DIV_WARMSTART;
    static int ENABLE_COUNT_NEIGHBOR_DOF;
#ifdef BACKWARD
    static int CURRENT_X_RB;
    static int FINAL_X_RB;
    static int FINAL_ANGLE_RB;
    static int INIT_V_RB;
    static int GRAD_INIT_V_RB;
    static int INIT_OMEGA_RB;
    static int GRAD_INIT_OMEGA_RB;
    static int TARGET_X;
    static int TARGET_ANGLE;
    static int TARGET_TIME;
    static int LOSS;
    static int LR;
    static int N_ITER;
    static int DEBUG;
    static int OPTIMIZE_ROTATION;
    static int TARGET_QUATERNION;
    static int FINAL_QUATERNION_RB;
    static int CURRENT_OMEGA_RB;
    static int CURRENT_V_RB;
    static int TOTAL_MASS_RB;
    static int TOTAL_MASS_FLUID;
    static int UNIFORM_ACC_RB_TIME;

    FORCE_INLINE bool isTrajectoryFinished() const{
      return m_is_trajectory_finished_callback;
    }
    FORCE_INLINE unsigned int get_step_count() const {
      return step_count;
    }

    void set_loss(const Real);
    Real get_loss();
    void set_loss_x(const Real);
    Real get_loss_x();
    void set_loss_rotation(const Real);
    Real get_loss_rotation();

    Real get_lr();
    void set_lr(const Real);

    void set_init_v_rb(unsigned int, const Vector3r);
    void set_init_omega_rb(unsigned int, const Vector3r);
    void set_init_omega_rb_to_joint(unsigned int, const Vector3r);

    BoundaryModel_Akinci2012* get_boundary_model_Akinci12(unsigned int);
    // ----------------------------------------------------------------------------------
    Vector3r get_init_v_rb(unsigned int);
    Vector3r get_init_omega_rb(unsigned int);

    Vector3r get_target_x(unsigned int);
    Vector3r get_init_x(unsigned int);
    void set_target_x(unsigned int, const Vector3r);
    Vector3r get_target_angle_in_radian(unsigned int);
    Vector3r get_target_angle_in_degree(unsigned int);
    Vector4r get_target_quaternion_vec4(unsigned int);

    bool is_trajectory_finish_callback();
    void clear_all_callbacks();
    bool is_in_new_trajectory();
    void set_in_new_trajectory(bool b);
    void reset_gradient(); // reset gradient for short horizon
    
    unsigned int get_num_1ring_fluid_particle(); 

    void set_custom_log_message(const std::string&);
    string get_custom_log_message();
    void add_log(const std::string&);

    virtual void beginStep();
    virtual void endStep();
    virtual void backwardPerStep();

    // ----------------------------------------------------------------------------------
    void setlbfgsMaxIter(unsigned int);
    void setlbfgsMemorySize(unsigned int);
    void setlbfgsLineSearchMethod(unsigned int);
    void setlbfgsMaxLineSearch(unsigned int);

    void setlbfgsFunction(LBFGSFuncWrapper::FuncType const& func)
    {
      lbfgs_wrapper.setFunction(func);
    }

    void setlbfgsInitPoint(VectorXr& initPoint);
    void setlbfgsUseNormalizedGrad(bool);

    void startlbfgsTraining();
    // ----------------------------------------------------------------------------------
#endif

    TimeStepDiffDFSPH();
    virtual ~TimeStepDiffDFSPH(void);

    virtual void step();
    virtual void reset();
		void countNeighborDOF();

    virtual void resize();
  };
}
