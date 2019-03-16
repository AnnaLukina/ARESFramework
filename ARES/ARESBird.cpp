#include "ARESBird.h"

ARESBird::ARESBird():
	position_(0, 0),
	velocity_(0, 0),
	acceleration_(0, 0),
	current_lower_bound_(0),
	current_upper_bound_(0)
{
}

ARESBird::ARESBird(const ARESParameters &parameters, const ARESPointCartesian &position, const ARESPointCartesian &velocity, const ARESPointCartesian &acceleration) :
	position_(position),
	velocity_(velocity),
	acceleration_(acceleration),
	current_lower_bound_(parameters.random_lower_bound),
	current_upper_bound_(parameters.random_upper_bound)
{}

//note that acceleration might change due to trimming :o
//problematic - this function move AND sets acceleration - check if this can be decomposed
void ARESBird::move(const ARESParameters &parameters, const ARESPointSpherical &input_spherical_acceleration)
{
	acceleration_ = input_spherical_acceleration;

	//#update velocities
	velocity_.x_ += (acceleration_.x_ * parameters.delta_t);
	velocity_.y_ += (acceleration_.y_ * parameters.delta_t);

	//update positions
	position_.x_ += (velocity_.x_ * parameters.delta_t);
	position_.y_ += (velocity_.y_ * parameters.delta_t);
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