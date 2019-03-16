#ifndef ARES_FLOCK_H_
#define ARES_FLOCK_H_

#include "ARESParameters.h"
#include "ARESBird.h"
#include "ARESParticle.h"

class ARESFlock
{

public:

	ARESBird *birds_;
	int num_birds_;

	ARESFlock(ARESParameters &parameters);
	void updatePositions(ARESParameters &parameters, ARESParticle &particle);	
	~ARESFlock();
};

#endif // !ARES_FLOCK_H_
