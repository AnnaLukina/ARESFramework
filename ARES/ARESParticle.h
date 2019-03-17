#ifndef ARES_PARTICLE_H_
#define ARES_PARTICLE_H_

#include "ARESParameters.h"
#include "ARESPointCartesian.h"
#include "ARESParticleElement.h"

class ARESFlock;

class ARESParticle {
public:
	int num_elements_;
	ARESParticleElement *elements_;
	//ARESParticleElement *personal_best_elements_;
	//double personal_best_objective_;

	ARESParticle();
	ARESParticle(const ARESParameters &parameters);
	
	void randomise(const ARESFlock &flock);

	void evolve(const ARESParameters &parameters, const ARESParticle &reference_particle, const ARESFlock &flock);

	ARESPointSpherical getAcceleration(int i);

	void initialise(const ARESParameters &parameters);

	ARESParticle& operator=(const ARESParticle &p);
	 
	~ARESParticle();
};

#endif // !ARES_PARTICLE_H_