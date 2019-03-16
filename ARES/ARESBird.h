#ifndef ARES_BIRD_H_
#define ARES_BIRD_H_

#include "ARESParameters.h"
#include "ARESPointCartesian.h"
#include "ARESPointSpherical.h"

class ARESBird
{	
public:

	ARESPointCartesian position_, velocity_, acceleration_;
	double current_lower_bound_, current_upper_bound_;

	ARESBird();
	ARESBird(const ARESParameters &parameters, const ARESPointCartesian &position, const ARESPointCartesian &velocity, const ARESPointCartesian &acceleration);
	
	//note that acceleration might change due to trimming :o
	//problematic - this function move AND sets acceleration - check if this can be decomposed
	void move(const ARESParameters &parameters, const ARESPointSpherical &input_spherical_acceleration);

	ARESPointSpherical createNewAcceleration() const;

	static ARESBird createRandomBird(const ARESParameters &parameters);

};


#endif // !ARES_BIRD
