#include "ARESSwarm.h"
#include <algorithm>

Swarm::Swarm(const ARESParameters & parameters, const ARESFlock &flock)
{
	assert(parameters.number_of_agents == flock.num_birds_);
	assert(parameters.number_of_agents > 0);
	
	counter_inertia_ = 0;
	
	num_particles_allocated_ = parameters.number_of_particles * parameters.horizon;
	num_particles_active_ = parameters.number_of_particles;
	
	particles_ = new ARESParticle[num_particles_allocated_];
	objective_particles_current_ = new double[num_particles_allocated_];
	
	particles_best_personal_ = new ARESParticle[num_particles_allocated_];
	objective_best_personal_ = new double[num_particles_allocated_];

	for (int i = 0; i < num_particles_allocated_; i++)
	{
		particles_[i].initialise(parameters);
		particles_best_personal_[i].initialise(parameters);	
	}

	for (int i = 0; i < num_particles_active_; i++)
	{
		particles_[i].randomise(flock);
		particles_best_personal_[i] = particles_[i];
		
		objective_particles_current_[i] = evaluator_.computeObjectiveForParticle(parameters, particles_[i], flock);
		objective_best_personal_[i] = objective_particles_current_[i];
	}
	
	particle_best_global_ = particles_[0];
	objective_best_global_ = objective_best_personal_[0];
	for (int i = 1; i < num_particles_active_; i++)
	{
		if (objective_best_global_ > objective_best_personal_[i])
		{
			objective_best_global_ = objective_best_personal_[i];
			particle_best_global_ = particles_[i];
		}
	}
}

void Swarm::evolveParticles(ARESParameters & parameters, const ARESFlock & flock)
{
	for (int i = 0; i < num_particles_active_; i++)
	{
		//evolve particles, include all the book-keeping
		particles_[i].evolve(parameters, particle_best_global_, flock);
		//update local information
		updateLocalInformation(parameters, flock, i);
		updateGlobalInformation(parameters, flock, particles_[i]);
		//TODO if the algorithms finds a better solution, should the algorithm consider the new best solution?
	}
}

void Swarm::updateLocalInformation(const ARESParameters &parameters, const ARESFlock &flock, int index_particle)
{
	objective_particles_current_[index_particle] = evaluator_.computeObjectiveForParticle(parameters, particles_[index_particle], flock); //todo verify flock is correct, since it seems the method relies on flock not being modified outside the method
	
	if (objective_particles_current_[index_particle] < objective_best_personal_[index_particle])
	{
		objective_best_personal_[index_particle] = objective_particles_current_[index_particle];
		particles_best_personal_[index_particle] = particles_[index_particle];
	}
}

void Swarm::updateGlobalInformation(ARESParameters &parameters, const ARESFlock &flock, const ARESParticle &candidate_particle)
{
	//update global best particle
	double candidate_objective_value = evaluator_.computeObjectiveForParticle(parameters, candidate_particle, flock);
	if (candidate_objective_value < objective_best_global_)
	{
		particle_best_global_ = candidate_particle;
		objective_best_global_ = candidate_objective_value;
		counter_inertia_ = std::max(counter_inertia_ - 1, 0);
	}
	else
	{
		counter_inertia_++;
	}
	//todo think about counter_inertia...it's a bit unusual
	//todo think about parameters getting changed during the algorithm...that's unusual

	//update initial_inertia
	if (counter_inertia_ < 2)
	{
		//todo should this be inertial or really counter_inertia?
		counter_inertia_ = std::max(parameters.inertia_min, std::min(parameters.inertia_max, 2.0 * parameters.current_inertia));
	}
	else if (counter_inertia_ > 5)
	{
		parameters.current_inertia = std::max(parameters.inertia_min, std::min(parameters.inertia_max, 0.5 * parameters.current_inertia));
	}
}

Swarm::~Swarm()
{
	delete[] particles_;
	delete[] particles_best_personal_;
	delete[] objective_best_personal_;
	delete[] objective_particles_current_;
}