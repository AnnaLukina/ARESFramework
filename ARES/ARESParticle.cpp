#include "ARESParticle.h"
#include "ARESFlock.h"

ARESParticle::ARESParticle() :
	num_elements_(0),
	elements_(0)
{
}

ARESParticle::ARESParticle(const ARESParameters &parameters):
	num_elements_(0),
	elements_(0)
{
	initialise(parameters);
}

void ARESParticle::randomise(const ARESFlock &flock)
{
	for (int i = 0; i < num_elements_; i++)
	{
		double lb = flock.birds_[i].current_lower_bound_;
		double ub = flock.birds_[i].current_upper_bound_;
		elements_[i].randomise(lb, ub);
		//personal_best_elements_[i] = elements_[i];
	}
	//TODO stopped here - need to see who computes the objective??
	//the algorithm I think
}

ARESPointSpherical ARESParticle::getAcceleration(int i)
{
	return elements_[i].acceleration_;
}

void ARESParticle::initialise(const ARESParameters &parameters)
{
	assert(elements_ == 0); // for now we only call this when the elements are null

	num_elements_ = parameters.number_of_particles;
	elements_ = new ARESParticleElement[num_elements_];
	//personal_best_elements_ = new ARESParticleElement[num_elements_];
	//personal_best_objective_ = 999999999;
}

ARESParticle & ARESParticle::operator=(const ARESParticle & p)
{
	assert(num_elements_ == p.num_elements_);
	for (int i = 0; i < num_elements_; i++)
	{
		elements_[i] = p.elements_[i];
	}
	return *this;
}

ARESParticle::~ARESParticle()
{
	delete[] elements_;
}

void ARESParticle::evolve(const ARESParameters &parameters, const ARESParticle &reference_particle, const ARESFlock &flock)
{
	assert(num_elements_ == reference_particle.num_elements_);
	assert(num_elements_ == parameters.number_of_agents);

	for (int i = 0; i < num_elements_; i++)
	{
		double lb = flock.birds_[i].current_lower_bound_;
		double ub = flock.birds_[i].current_upper_bound_;
		elements_[i].evolve(parameters, reference_particle.elements_[i], lb, ub);
		//update personal best??
	}
}