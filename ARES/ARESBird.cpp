#include "ARESBird.h"

ARESBird::ARESBird(const ARESParameters &parameters, const ARESPointCartesian &position, const ARESPointCartesian &velocity, const ARESPointCartesian &acceleration) :
	position_(position),
	velocity_(velocity),
	acceleration_(acceleration),
	current_lower_bound_(parameters.random_lower_bound),
	current_upper_bound_(parameters.random_upper_bound)
{}

//note that acceleration might change due to trimming :o
//problematic - this function move AND sets acceleration - check if this can be decomposed
void ARESBird::move(const ARESParameters &parameters, ARESPointSpherical &input_spherical_acceleration)
{
	//TODO - what? Is this correct? I think this is a bug -> should be acceleration not velocity that becomes too big

	//trim - reduce the acceleration if it is too big
	//cannot increase the velocity abruptly - todo this is a big strange though, the way its computed, especially if there's a delta_t
	double velocity_norm = velocity_.norm();
	if (input_spherical_acceleration.magnitude_ >= velocity_norm)
	{
		input_spherical_acceleration.magnitude_ = velocity_norm;
	}

	acceleration_ = input_spherical_acceleration;

	//#update velocities
	velocity_.x_ += (acceleration_.x_ * parameters.delta_t);
	velocity_.y_ += (acceleration_.y_ * parameters.delta_t);

	//update positions
	position_.x_ += (velocity_.x_ * parameters.delta_t);
	position_.y_ += (velocity_.y_ * parameters.delta_t);
}

ARESPointSpherical ARESBird::createNewAcceleration() const
{
	double magnitude = ARESLibrary::random_uniform(current_lower_bound_, current_upper_bound_);
	double angle = ARESLibrary::random_uniform(current_lower_bound_, current_upper_bound_);

	return ARESPointSpherical(magnitude, angle);
}

ARESBird ARESBird::createRandomBird(const ARESParameters &parameters)
{
	return ARESBird(
		parameters,
		ARESPointCartesian::createRandomPoint(0, parameters.bounding_box), //position
		ARESPointCartesian::createRandomPoint(0.25, 0.75), //velocity
		ARESPointCartesian(0, 0)
	);
}