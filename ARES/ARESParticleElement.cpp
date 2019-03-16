#include "ARESParticleElement.h"

#include <algorithm>

#include "ARESLibrary.h"

ARESParticleElement::ARESParticleElement():
	acceleration_(0, 0),
	personal_best_(0, 0),
	growth_(0, 0)
{
}

ARESParticleElement::ARESParticleElement(double lower_bound, double upper_bound):
	ARESParticleElement()
{
	randomise(lower_bound, upper_bound);
}

void ARESParticleElement::randomise(double lower_bound, double upper_bound)
{
	acceleration_ = ARESPointSpherical::createRandomPoint(lower_bound, upper_bound);
	personal_best_ = acceleration_;
	growth_ = ARESPointSpherical::createRandomPoint(lower_bound - upper_bound, upper_bound - lower_bound);//note that both components are not identical

}

void ARESParticleElement::evolve(const ARESParameters &parameters, const ARESParticleElement &reference_element, double lower_bound, double upper_bound)
{
	evolveGrowth(parameters, reference_element);
	updateAcceleration(growth_.magnitude_, growth_.angle_);
	nullifyGrowthConditional(lower_bound, upper_bound);
	clampAcceleration(lower_bound, upper_bound);
}

void ARESParticleElement::evolveGrowth(const ARESParameters &parameters, const ARESParticleElement &reference_element)
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

double ARESParticleElement::computeGrowthComponentEvolution(const ARESParameters &parameters, double growth_component, double current_component, double personal_best_component, double reference_best_component)
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

void ARESParticleElement::updateAcceleration(double delta_magnitude, double delta_angle)
{
	acceleration_.magnitude_ += delta_magnitude;
	acceleration_.angle_ += delta_angle;
}

void ARESParticleElement::nullifyGrowthConditional(double lower_bound, double upper_bound)
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

void ARESParticleElement::clampAcceleration(double lower_bound, double upper_bound)
{
	//clamp values if they are too large
	acceleration_.magnitude_ = std::min(acceleration_.magnitude_, upper_bound);
	acceleration_.magnitude_ = std::max(acceleration_.magnitude_, lower_bound);

	acceleration_.angle_ = std::min(acceleration_.angle_, upper_bound);
	acceleration_.angle_ = std::max(acceleration_.angle_, lower_bound);
}