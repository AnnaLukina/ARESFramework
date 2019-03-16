#ifndef ARES_H_
#define ARES_H_

//todo make it general
class ARES
{
	void run_experiment(int num_birds)
	{
		//generate initial positions
		//	current_objective_value = computeObjective(parameters = parameters, flock = current_flock)
		// 	best_particles = []

		/*time_start = time.time()
			for time_i in range(parameters.number_of_time_steps) :
				#todo properly include the horizon approach and levels
				print(time_i)
				print("current objective value: {}".format(current_objective_value))
				(best_particle, best_objective_value) = computeBestParticle(flock = current_flock, parameters = parameters)
				while best_objective_value > current_objective_value:
		(best_particle, best_objective_value) = computeBestParticle(flock = current_flock, parameters = parameters)
			parameters.horizon += 1 # explore longer horizons until a better solution is found
			if parameters.horizon > 10:
		print("Horizon reached its limit")
			break
			parameters.horizon = 1 # reset horizon
			current_flock.updatePositions(parameters = parameters, particle = best_particle)
			current_objective_value = best_objective_value
			best_particles.append(best_particle)

			# update bounds on accelerations for the next steps
			for bird in current_flock.birds:
		bird.lower_bound = 0.0
			bird.upper_bound = np.linalg.norm([bird.x_velocity, bird.y_velocity])

			if current_objective_value <= 0.05 :
				print("best objective: {}\n".format(current_objective_value))
				print("mama mia it werks!\n")
				break

			//	print("Time passed = {}\n".format(time.time() - time_start))
			//display pic

				*/
	}
};

#endif // !ARES_H_
