#ifndef ARES_PARTICLE_ELEMENT_H_
#define ARES_PARTICLE_ELEMENT_H_

#include <algorithm>

#include "ARESBird.h"
#include "ARESPointSpherical.h"
#include "ARESLibrary.h"

class ParticleElement {

public:

	ARESPointSpherical acceleration_;
	ARESPointSpherical personal_best_;
	ARESPointSpherical growth_;

	ParticleElement(const ARESBird &bird) :
		acceleration_(bird.createNewAcceleration()),
		personal_best_(acceleration_),
		growth_(ARESPointSpherical::createRandomPoint(bird.current_lower_bound_ - bird.current_upper_bound_, bird.current_upper_bound_ - bird.current_lower_bound_)) //note that both components are not identical
	{
	}

	void evolve(const ARESParameters &parameters, const ParticleElement &reference_element, const ARESBird &bird)
	{
		evolveGrowth(parameters, reference_element);		
		updateAcceleration(growth_.magnitude_, growth_.angle_);
		nullifyGrowthConditional(bird.current_lower_bound_, bird.current_upper_bound_);
		clampAcceleration(bird.current_lower_bound_, bird.current_upper_bound_);
	}

private:
	void evolveGrowth(const ARESParameters &parameters, const ParticleElement &reference_element)
	{
		//evolve growth magnitude
		growth_.magnitude_ = computeGrowthComponentEvolution(
			parameters,
			growth_.magnitude_,
			acceleration_.magnitude_,
			personal_best_.magnitude_,
			reference_element.personal_best_.magnitude_
		);

		//evolve growth angle
		growth_.angle_ = computeGrowthComponentEvolution(
			parameters,
			growth_.angle_,
			acceleration_.angle_,
			personal_best_.angle_,
			reference_element.personal_best_.angle_
		);
	}

	double computeGrowthComponentEvolution(const ARESParameters &parameters, double growth_component, double current_component, double personal_best_component, double reference_best_component)
	{
		//# f1 = (1 - (self.K_i - self.K_max) / self.K_max) * np.random.uniform(0, 1, 1)
		//# f2 = (self.K_i - self.K_min) / self.K_max * np.random.uniform(0, 1, 1)
		//# todo update the comment above - what is this lel

		double a = parameters.inertia * growth_component;
		double b = parameters.cognitive_coefficient * (personal_best_component - current_component);
		b *= ARESLibrary::random_uniform(0, 1);
		double c = parameters.social_coefficient * (reference_best_component - current_component);
		c *= ARESLibrary::random_uniform(0, 1);
		return a + b + c;
	}

	void updateAcceleration(double delta_magnitude, double delta_angle)
	{
		acceleration_.magnitude_ += delta_magnitude;
		acceleration_.angle_ += delta_angle;
	}
		
	void nullifyGrowthConditional(double lower_bound, double upper_bound)
	{
		if (ARESLibrary::isOutOfBounds(acceleration_.magnitude_, lower_bound, upper_bound))
		{
			growth_.magnitude_ = 0.0;
		}

		if (ARESLibrary::isOutOfBounds(acceleration_.angle_, lower_bound, upper_bound))
		{
			growth_.angle_ = 0.0;
		}	
	}

	void clampAcceleration(double lower_bound, double upper_bound)
	{
		//clamp values if they are too large
		acceleration_.magnitude_ = std::min(acceleration_.magnitude_, upper_bound);
		acceleration_.magnitude_ = std::max(acceleration_.magnitude_, lower_bound);

		acceleration_.angle_ = std::min(acceleration_.angle_, upper_bound);
		acceleration_.angle_ = std::max(acceleration_.angle_, lower_bound);
	}
	
};

#endif // !ARES_PARTICLE_ELEMENT_H_
