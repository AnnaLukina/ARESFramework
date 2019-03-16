#include "ARESFlock.h"

ARESFlock::ARESFlock(ARESParameters &parameters)
{
	num_birds_ = parameters.number_of_agents;
	birds_ = new ARESBird[num_birds_];
	for (int i = 0; i < num_birds_; i++)
	{
		birds_[i] = ARESBird::createRandomBird(parameters);
	}
}

void ARESFlock::updatePositions(ARESParameters &parameters, ARESParticle &particle)
{
	assert(num_birds_ == particle.num_elements_);

	for (int i = 0; i < num_birds_; i++)
	{
		birds_[i].move(parameters, particle.getAcceleration(i));
	}
}

ARESFlock::~ARESFlock()
{
	delete[] birds_;
}
