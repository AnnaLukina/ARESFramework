#include "ARESPointSpherical.h"
#include "ARESLibrary.h"

//ARESPointSpherical::ARESPointSpherical(ARESPointSpherical(double magnitude, double angle)
//{

	//magnitude_ = ARESLibrary::random_uniform(bird.current_lower_bound_, bird.current_upper_bound_);
	//	angle_ = ARESLibrary::random_uniform(bird.current_lower_bound_, bird.current_upper_bound_);
//}

ARESPointSpherical::ARESPointSpherical(double magnitude, double angle):
	magnitude_(magnitude), angle_(angle)
{}

ARESPointSpherical ARESPointSpherical::createRandomPoint(double lower_bound, double upper_bound)
{
	return ARESPointSpherical(ARESLibrary::random_uniform(lower_bound, upper_bound), ARESLibrary::random_uniform(lower_bound, upper_bound));
}
