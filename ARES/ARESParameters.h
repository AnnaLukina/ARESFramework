#ifndef ARES_PARAMETERS_H_
#define ARES_PARAMETERS_H_

#include "ARESLibrary.h"

class ARESParameters {

public:
	double delta_t;
	int horizon;
	
	int number_of_agents;
	double bounding_box;
	int number_of_time_steps;
	double random_upper_bound;
	double random_lower_bound;

	double cognitive_coefficient;
	double social_coefficient;
	double inertia_max;
	double inertia_min;
	double inertia;
	int number_of_swarm_steps;
	int number_of_particles;

	ARESParameters() {
		delta_t = 1.0;
		horizon = 1;

		//flock parameters
		number_of_agents = 10;
		bounding_box = 3.0;
		number_of_time_steps = 50;
		random_upper_bound = 2 * ARESLibrary::PI();
		random_lower_bound = 0.0;

		//swarm parameters
		cognitive_coefficient = 0.5;
		social_coefficient = 0.5;
		inertia_max = 1.0;
		inertia_min = 0.1;
		inertia = inertia_min;
		number_of_swarm_steps = 1000;
		number_of_particles = 1000;
	}
};
#endif // !
