#ifndef ARES_EVALUATOR_H_
#define ARES_EVALUATOR_H_

#include "ARESFlock.h"
#include "ARESParticle.h"
#include "ARESParameters.h"

class ARESEvaluator
{
public:
	double computeObjectiveForFlock(const ARESParameters &parameters, const ARESFlock &flock);
	double computeObjectiveForParticle(const ARESParameters &parameters, const ARESParticle &particle, const ARESFlock &flock);
};

#endif // !ARES_OBJECTIVE_COMPUTER_H_
