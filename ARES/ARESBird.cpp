#include "ARESBird.h"

ARESBird::ARESBird(ARESParameters &parameters, ARESPointCartesian &position, ARESPointCartesian &velocity, ARESPointCartesian &acceleration) :
	position_(position),
	velocity_(velocity),
	acceleration_(acceleration),
	lower_bound_(parameters.random_lower_bound),
	upper_bound_(parameters.random_upper_bound)
{}

//note that acceleration might change due to trimming :o
//problematic - this function move AND sets acceleration - check if this can be decomposed
void ARESBird::move(ARESParameters &parameters, ARESPointSpherical &input_spherical_acceleration)
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