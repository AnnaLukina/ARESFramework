#ifndef ARES_BIRD_H_
#define ARES_BIRD_H_

#include "ARESParameters.h"
#include "ARESPointCartesian.h"
#include "ARESPointSpherical.h"

class ARESBird
{	
public:

	ARESPointCartesian position_, velocity_, acceleration_;
	double lower_bound_, upper_bound_;

	ARESBird(ARESParameters &parameters, ARESPointCartesian &position, ARESPointCartesian &velocity, ARESPointCartesian &acceleration);
	
	//note that acceleration might change due to trimming :o
	//problematic - this function move AND sets acceleration - check if this can be decomposed
	void move(ARESParameters &parameters, ARESPointSpherical &input_spherical_acceleration);
};


#endif // !ARES_BIRD
