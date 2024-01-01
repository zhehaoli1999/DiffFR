#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <vector> 
#include "SPlisHSPlasH/Common.h"
#include "SPlisHSPlasH/FluidModel.h"

#define BACKWARD

using namespace std;

namespace SPH {

    class SimulationDataDiffDFSPH
    {
    public:
        SimulationDataDiffDFSPH();
        virtual ~SimulationDataDiffDFSPH(); 

		protected:	

			/** \brief factor \f$\alpha_i\f$ */
			std::vector<std::vector<Real>> m_factor;
			/** \brief stores \f$\kappa\f$ value of last time step for a warm start of the pressure solver */
			std::vector<std::vector<Real>> m_kappa; 
			/** \brief stores \f$\kappa^v\f$ value of last time step for a warm start of the divergence solver */
			std::vector<std::vector<Real>> m_kappaV;
			/** \brief advected density */
			std::vector<std::vector<Real>> m_density_adv;
#ifdef BACKWARD
      Real target_time, loss, learning_rate;
      Real loss_x, loss_rotation; 
      
      Real uniform_accelerate_rb_time_at_beginning; 
      vector<vector<Vector3r>> m_sum_grad_p_k_array;

      unsigned int m_opt_iter; // iteration count of optimization 
    
      unsigned int num_1ring_fluid_particle; 
    
      Real n_boundary_models; 
      vector<Vector3r> target_x; 
      vector<Vector3r> init_x; 
      vector<Vector3r> target_angle_in_radian; 
      vector<Vector3r> target_angle_in_degree; 
      vector<Quaternionr> target_quaternion;

      // --------------------------------------------------------------
      vector<Vector3r> init_v_rb; 
      vector<Vector3r> init_omega_rb;
    
      vector<Quaternionr> init_rb_rotation;
      
      vector<Vector3r> current_omega_rb;
      vector<Vector3r> current_v_rb;
      vector<Vector3r> current_x_rb;
      Real total_mass_rb;
      Real total_mass_fluid;

      string custom_log_msg; 

      friend class TimeStepDiffDFSPH; 
#endif 
        //vector<Real> timeStepArray;

      public:
			/** Initialize the arrays containing the particle data.
			*/
			virtual void init();

			/** Release the arrays containing the particle data.
			*/
			virtual void cleanup();

			/** Reset the particle data.
			*/
			virtual void reset();
			/** Important: First call m_model->performNeighborhoodSearchSort() 
			 * to call the z_sort of the neighborhood search.
			 */
			void performNeighborhoodSearchSort();
			void emittedParticles(FluidModel *model, const unsigned int startIndex);

			FORCE_INLINE const Real getFactor(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_factor[fluidIndex][i];
			}

			FORCE_INLINE Real& getFactor(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_factor[fluidIndex][i];
			}

			FORCE_INLINE void setFactor(const unsigned int fluidIndex, const unsigned int i, const Real p)
			{
				m_factor[fluidIndex][i] = p;
			}

			FORCE_INLINE const Real getKappa(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_kappa[fluidIndex][i];
			}

			FORCE_INLINE Real& getKappa(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_kappa[fluidIndex][i];
			}

			FORCE_INLINE void setKappa(const unsigned int fluidIndex, const unsigned int i, const Real p)
			{
				m_kappa[fluidIndex][i] = p;
			}

			FORCE_INLINE const Real getKappaV(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_kappaV[fluidIndex][i];
			}

			FORCE_INLINE Real& getKappaV(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_kappaV[fluidIndex][i];
			}

			FORCE_INLINE void setKappaV(const unsigned int fluidIndex, const unsigned int i, const Real p)
			{
				m_kappaV[fluidIndex][i] = p;
			}

			FORCE_INLINE const Real getDensityAdv(const unsigned int fluidIndex, const unsigned int i) const
			{
				return m_density_adv[fluidIndex][i];
			}

			FORCE_INLINE Real& getDensityAdv(const unsigned int fluidIndex, const unsigned int i)
			{
				return m_density_adv[fluidIndex][i];
			}

			FORCE_INLINE void setDensityAdv(const unsigned int fluidIndex, const unsigned int i, const Real d)
			{
				m_density_adv[fluidIndex][i] = d;
			}
#ifdef BACKWARD
        // need to add some read and set function 
		    FORCE_INLINE Vector3r& get_sum_grad_p_k(const unsigned int fluidIndex, const unsigned int i)
			  {
            //TODO: to begin with we only have one type of fluid
				    return m_sum_grad_p_k_array[fluidIndex][i];
			  }

		    FORCE_INLINE Vector3r& get_target_x(const unsigned bm_index)
			  {
          return target_x[bm_index]; 
			  }
		    FORCE_INLINE Vector3r& get_init_x(const unsigned bm_index)
			  {
          return init_x[bm_index]; 
			  }
		    FORCE_INLINE Vector3r& get_target_angle_in_radian(const unsigned bm_index)
			  {
          return target_angle_in_radian[bm_index]; 
			  }
		    FORCE_INLINE Vector3r& get_target_angle_in_degree(const unsigned bm_index)
			  {
          target_angle_in_degree[bm_index] = target_angle_in_radian[bm_index] / M_PI * 180; 
          return target_angle_in_degree[bm_index]; 
			  }
		    FORCE_INLINE Quaternionr& get_target_quaternion(const unsigned bm_index)
			  {
          auto ai =  target_angle_in_radian[bm_index];
          #ifdef USE_DOUBLE
          target_quaternion[bm_index] = Eigen::AngleAxisd(ai[0], Vector3d::UnitX())
                            *Eigen::AngleAxisd(ai[1], Vector3d::UnitY())
                            * Eigen::AngleAxisd(ai[2], Vector3d::UnitZ());
          #else
          target_quaternion[bm_index] = Eigen::AngleAxisf(ai[0], Vector3f::UnitX())
                            *Eigen::AngleAxisf(ai[1], Vector3f::UnitY())
                            * Eigen::AngleAxisf(ai[2], Vector3f::UnitZ());
          #endif
          // So actually the target_quaternion is just an "increment quaternion", not the final one? this is not easy to use 
          //target_quaternion[bm_index] = target_quaternion[bm_index] * get_init_rb_rotation(bm_index);  
      
          return target_quaternion[bm_index]; 
			  }

		    FORCE_INLINE Vector3r& get_init_v_rb(const unsigned bm_index)
			  {
          return init_v_rb[bm_index]; 
			  }
		    FORCE_INLINE Vector3r& get_init_omega_rb(const unsigned bm_index)
			  {
          return init_omega_rb[bm_index]; 
			  }

        // ----------------------------------------------------------------  

		    FORCE_INLINE Real& get_loss()
			  {
          return loss; 
			  }
		    FORCE_INLINE Real& get_loss_x()
			  {
          return loss_x; 
			  }
		    FORCE_INLINE Real& get_loss_rotation()
			  {
          return loss_rotation; 
			  }

		    FORCE_INLINE Real& get_learning_rate()
			  {
          return learning_rate; 
			  }
		    FORCE_INLINE Real& get_target_time()
			  {
          return target_time; 
			  }
		    FORCE_INLINE Real& get_uniform_accelerate_rb_time_at_beginning()
			  {
          return uniform_accelerate_rb_time_at_beginning; 
			  }

		    FORCE_INLINE unsigned int& get_opt_iter()
			  {
          return m_opt_iter; 
			  }
		    FORCE_INLINE unsigned int increase_opt_iter()
			  {
          return m_opt_iter++; 
			  }

        FORCE_INLINE Quaternionr& get_init_rb_rotation(const unsigned bm_index)
        {
          return init_rb_rotation[bm_index];
        }

		    FORCE_INLINE Vector3r& get_current_omega_rb(const unsigned bm_index)
			  {
          return current_omega_rb[bm_index]; 
			  }
		    FORCE_INLINE Vector3r& get_current_v_rb(const unsigned bm_index)
			  {
          return current_v_rb[bm_index]; 
			  }
		    FORCE_INLINE Vector3r& get_current_x_rb(const unsigned bm_index)
			  {
          return current_x_rb[bm_index]; 
			  }

		    FORCE_INLINE Real& get_total_mass_rb()
			  {
          return total_mass_rb; 
			  }
		    FORCE_INLINE Real& get_total_mass_fluid()
			  {
          return total_mass_fluid; 
			  }
		    FORCE_INLINE string& get_custom_log_message()
			  {
          return custom_log_msg; 
			  }
        FORCE_INLINE void clear_custom_log_message()
       {
          custom_log_msg.clear();
        }
#endif 

    }; 
}
