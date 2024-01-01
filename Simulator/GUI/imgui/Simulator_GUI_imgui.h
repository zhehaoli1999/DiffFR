#ifndef __Simulator_GUI_imgui_h__
#define __Simulator_GUI_imgui_h__

#include "SPlisHSPlasH/Common.h"
#include "../Simulator_GUI_Base.h"
#include <vector>

namespace SPH 
{	
	class Simulator_GUI_imgui : public Simulator_GUI_Base
	{
		struct Target_GUI_info
		{
			std::vector<Vector3f> transformed_vertices;
			std::vector<Vector3f> transformed_axis;
		};

		public:
			Simulator_GUI_imgui(SimulatorBase *base);
			virtual ~Simulator_GUI_imgui();
			static std::vector<Target_GUI_info> &get_target_infos()
			{
				static std::vector<Target_GUI_info> target_infos;
				return target_infos;
			}

		protected:
			unsigned int m_currentFluidModel;
			Vector3r m_oldMousePos;
			std::vector<std::string> m_colorFieldNames;
			std::vector<std::vector<unsigned int>> m_selectedParticles;			

			std::vector<std::vector<unsigned int>>& getSelectedParticles() { return m_selectedParticles; }
			void initImgui();
			void initImguiParameters();
			void renderBoundary();
			void renderJoints();
			void particleInfo();

			static void selection(const Vector2i &start, const Vector2i &end, void *clientData);
			static void mouseMove(int x, int y, void *clientData);

			void switchPause();
			static void switchDrawMode();
				
			void destroy();

		public:
			virtual void init(int argc, char **argv, const char *name);
			virtual void initParameterGUI();
			virtual void initSimulationParameterGUI();
			virtual void render();
			virtual void reset();
			virtual void update();
			virtual void cleanup();
			virtual void run();
			virtual void show();
			virtual void endShow();
			virtual void stop();
			virtual void addKeyFunc(char k, std::function<void()> const& func);

			void createSimulationParameterGUI();
	};
}

#endif
