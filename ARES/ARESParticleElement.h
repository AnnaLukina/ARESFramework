#ifndef ARES_PARTICLE_ELEMENT_H_
#define ARES_PARTICLE_ELEMENT_H_

#include "ARESBird.h"
#include "ARESPointSpherical.h"

class ARESParticleElement {

public:

	ARESPointSpherical acceleration_;
	ARESPointSpherical personal_best_;
	ARESPointSpherical growth_;

	ARESParticleElement();
	ARESParticleElement(double lower_bound, double upper_bound);

	void randomise(double lower_bound, double upper_bound);

	void evolve(const ARESParameters &parameters, const ARESParticleElement &reference_element, double lower_bound, double upper_bound);

private:
	void evolveGrowth(const ARESParameters &parameters, const ARESParticleElement &reference_element);

	double computeGrowthComponentEvolution(const ARESParameters &parameters, double growth_component, double current_component, double personal_best_component, double reference_best_component);
	void updateAcceleration(double delta_magnitude, double delta_angle);
	void nullifyGrowthConditional(double lower_bound, double upper_bound);
	void clampAcceleration(double lower_bound, double upper_bound);
	
};

#endif // !ARES_PARTICLE_ELEMENT_H_
