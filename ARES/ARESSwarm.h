#ifndef ARES_SWARM_H
#define ARES_SWARM_H

#include "ARESParameters.h"
#include "ARESFlock.h"

class Swarm 
{
public:

	ARESParticles * particles_;
	
	Swarm(const ARESParameters &parameters);

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
	
	~Swarm()
	{
		delete[] particles_;
	}

};

/*
class Swarm :
	def __init__(self, parameters, flock) :
	self.particles = [Particle(parameters = parameters, flock = flock) for i in range(parameters.number_of_particles)]

	p = min(self.particles, key = lambda p : computeObjectiveForParticle(parameters = parameters, flock = flock, particle = p))
	self.global_best_particle = copy.deepcopy(p)
	self.global_best_objective_value = computeObjectiveForParticle(parameters = parameters, flock = flock, particle = self.global_best_particle)

	self.inertia_counter = 0

	def evolveParticles(self, parameters, flock) :
	for particle in self.particles :
		particle.evolve(parameters = parameters, flock = flock, reference_particle = self.global_best_particle)
		self.updateGlobalInformation(parameters = parameters, flock = flock, candidate_particle = particle)

		def updateGlobalInformation(self, parameters, flock, candidate_particle) :
		#update global best partical
		candidate_objective_value = computeObjectiveForParticle(parameters = parameters, flock = flock, particle = candidate_particle)
		if candidate_objective_value < self.global_best_objective_value:
self.global_best_particle = copy.deepcopy(candidate_particle)
self.global_best_objective_value = candidate_objective_value
self.inertia_counter = max(self.inertia_counter - 1, 0)
		else:
self.inertia_counter += 1

# update inertia
if self.inertia_counter < 2:
self.inertia_counter = max(parameters.inertia_min, min(parameters.inertia_max, 2.0 * parameters.inertia))
else:
if self.inertia_counter > 5:
parameters.inertia = max(parameters.inertia_min, min(parameters.inertia_max, 0.5 * parameters.inertia))
*/

#endif // !ARES_SWARM_H
