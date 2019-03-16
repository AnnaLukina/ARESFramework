#ifndef ARES_PARTICLE_H_
#define ARES_PARTICLE_H_

#include "ARESParameters.h"
#include "ARESPointCartesian.h"
#include "ARESParticleElement.h"
#include "ARESFlock.h"

class ARESParticle {
public:
	int num_elements_;
	ARESParticleElement *elements_;
	ARESParticleElement *personal_best_elements_;
	double personal_best_objective_;

	ARESParticle(const ARESParameters &parameters)
	{
		num_elements_ = parameters.number_of_particles;
		elements_ = new ARESParticleElement[num_elements_];
		personal_best_elements_ = new ARESParticleElement[num_elements_];
		personal_best_objective_ = 999999999;
	}

	void randomise(const ARESFlock &flock)
	{
		for (int i = 0; i < num_elements_; i++)
		{
			double lb = flock.birds_[i].current_lower_bound_;
			double ub = flock.birds_[i].current_upper_bound_;
			elements_[i].randomise(lb, ub);
			personal_best_elements_[i] = elements_[i];
		}
		//TODO stopped here - need to see who computes the objective??
	}

	ARESPointSpherical getAcceleration(int i);
	
	//ARESParticle(ARESParameters &parameters)
};

/*
class Particle{
	Particle(ARESParameters &parameters, flock) :
	self.elements = [ARESParticleElement(bird = flock.birds[i]) for i in range(parameters.number_of_agents)]

	self.personal_best_elements_ = copy.deepcopy(self.elements)
	self.personal_best_objective_value = computeObjectiveForParticle(parameters = parameters, flock = flock, particle = self)


	def evolve(self, parameters, flock, reference_particle) :
	assert(parameters.number_of_agents == len(self.elements))

	for i in range(parameters.number_of_agents) :
		self.elements[i].evolve(parameters = parameters, reference_element = reference_particle.elements[i], bird = flock.birds[i])
		self.updatePersonalBestElements(flock = flock, candidate_elements = self.elements)

		def updatePersonalBestElements(self, flock, candidate_elements) :
		candidate_objective_value = computeObjectiveForParticle(parameters = parameters, flock = flock, particle = self)
		if candidate_objective_value < self.personal_best_objective_value :
			self.personal_best_elements_ = copy.deepcopy(candidate_elements)
			self.personal_best_objective_value = candidate_objective_value

			def getAcceleration(self, agent_i) :
			return self.elements[agent_i].acceleration

};

*/

#endif // !ARES_PARTICLE_H_
