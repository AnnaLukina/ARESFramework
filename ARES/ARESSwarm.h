#ifndef ARES_SWARM_H
#define ARES_SWARM_H

#include "ARESParameters.h"
#include "ARESFlock.h"
#include "ARESParticle.h"
#include "ARESEvaluator.h"

class Swarm		
{
public:

	int counter_inertia_;

	int num_particles_active_;
	int num_particles_allocated_;

	ARESParticle * particles_;

	double *objective_particles_current_;	
	ARESParticle * particles_best_personal_;
	double * objective_best_personal_;

	ARESParticle particle_best_global_;
	double objective_best_global_;
	
	ARESEvaluator evaluator_;

	Swarm(const ARESParameters &parameters, const ARESFlock &flock);

	void evolveParticles(ARESParameters &parameters, const ARESFlock &flock);
	void updateLocalInformation(const ARESParameters &parameters, const ARESFlock &flock, int index_particle);
	void updateGlobalInformation(ARESParameters &parameters, const ARESFlock &flock, const ARESParticle &candidate_particle);

	void randomiseParticles(ARESFlock &flock);

	void trimAcceleration()
	{
		/*
		//TODO - what? Is this correct? I think this is a bug -> should be acceleration not velocity that becomes too big
	//trim - reduce the acceleration if it is too big
	//cannot increase the velocity abruptly - todo this is a big strange though, the way its computed, especially if there's a delta_t
	double velocity_norm = velocity_.norm();
	if (input_spherical_acceleration.magnitude_ >= velocity_norm)
	{
		input_spherical_acceleration.magnitude_ = velocity_norm;
	}
		*/
	}
	
	~Swarm();

};

/*
class Swarm :
	
	
		def updateGlobalInformation(self, parameters, flock, candidate_particle) :
		#update global best partical
		candidate_objective_value = computeObjectiveForParticle(parameters = parameters, flock = flock, particle = candidate_particle)
		if candidate_objective_value < self.global_best_objective_value:
self.global_best_particle = copy.deepcopy(candidate_particle)
self.global_best_objective_value = candidate_objective_value
self.inertia_counter = max(self.inertia_counter - 1, 0)
		else:
self.inertia_counter += 1

# update initial_inertia
if self.inertia_counter < 2:
self.inertia_counter = max(parameters.inertia_min, min(parameters.inertia_max, 2.0 * parameters.initial_inertia))
else:
if self.inertia_counter > 5:
parameters.initial_inertia = max(parameters.inertia_min, min(parameters.inertia_max, 0.5 * parameters.initial_inertia))
*/

#endif // !ARES_SWARM_H