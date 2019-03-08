import numpy as np
import copy

class parameters:
    def __init__(self):
        #flock parameters
        self.number_of_agents = 7
        self.bounding_box = 3.0
        self.number_of_time_steps = 100
        self.random_upper_bound = 2*np.pi
        self.random_lower_bound = 0.0

        #swarm parameters
        self.cognitive_coefficient = 1 / 2
        self.social_coefficient = 1 / 2
        self.inertia_max = 1.0
        self.inertia_min = 0.1
        self.inertia = self.inertia_min
        self.number_of_swarm_steps = 100
        self.number_of_particles = 100

class Acceleration:

    def __init__(self, parameters):
        self.magnitude = np.random.uniform(parameters.random_lower_bound, parameters.random_upper_bound)
        self.angle = np.random.uniform(parameters.random_lower_bound, parameters.random_upper_bound)

class ParticleElement:
    def __init__(self, parameters):
        self.acceleration = Acceleration(parameters=parameters)

        self.personal_best_magnitude = self.acceleration.magnitude
        self.personal_best_angle = self.acceleration.angle

        self.magnitude_growth = np.random.uniform(parameters.random_lower_bound - parameters.random_upper_bound,
                                        parameters.random_upper_bound - parameters.random_lower_bound)

        self.angle_growth = np.random.uniform(parameters.random_lower_bound - parameters.random_upper_bound,
                                                  parameters.random_upper_bound - parameters.random_lower_bound)

    def evolve(self, parameters, reference_element):
        # f1 = (1 - (self.K_i - self.K_max) / self.K_max) * np.random.uniform(0,1,1)
        # f2 = (self.K_i - self.K_min) / self.K_max * np.random.uniform(0,1,1)
        # todo update the comment above

        #evolve magnitude
        a = parameters.inertia * self.magnitude_growth
        b = parameters.cognitive_coefficient * (self.personal_best_magnitude - self.acceleration.magnitude)
        b *= np.random.uniform(0, 1)
        c = parameters.social_coefficient * (reference_element.personal_best_magnitude - self.acceleration.magnitude )
        c *= np.random.uniform(0, 1)
        self.magnitude_growth =  a + b + c

        #evolve angle
        a = parameters.inertia * self.angle_growth
        b = parameters.cognitive_coefficient * (self.personal_best_angle - self.acceleration.angle)
        b *= np.random.uniform(0, 1)
        c = parameters.social_coefficient * (reference_element.personal_best_angle - self.acceleration.angle)
        c *= np.random.uniform(0, 1)

        self.angle_growth = a + b + c

        #update the values accordingly
        self.acceleration.magnitude += self.magnitude_growth
        self.acceleration.angle += self.angle_growth

        #note that these random bounds change during the execution of the algorithm, i.e. they are not fixed lmao
        if self.acceleration.magnitude > parameters.random_upper_bound or self.acceleration.magnitude < parameters.random_lower_bound:
            self.magnitude_growth = 0.0

        if self.acceleration.angle > parameters.random_upper_bound or self.acceleration.angle < parameters.random_lower_bound:
            self.angle_growth = 0.0

        #clamp values if they are too large
        self.acceleration.magnitude = min(self.acceleration.magnitude, parameters.random_upper_bound)
        self.acceleration.magnitude = max(self.acceleration.magnitude, parameters.random_lower_bound)

        self.acceleration.angle = min(self.acceleration.angle, parameters.random_upper_bound)
        self.acceleration.angle = max(self.acceleration.angle, parameters.random_lower_bound)

class Particle:
    def __init__(self, parameters, flock):
        self.elements = [ParticleElement(parameters=parameters) for i in range(parameters.number_of_agents)]

        self.personal_best_elements = copy.deepcopy(self.elements)
        self.personal_best_objective = computeObjectiveForParticle(flock=flock, particle=self)


    def evolve(self, parameters, flock, reference_particle):
        assert (parameters.number_of_agents == len(self.elements))

        for i in range(parameters.number_of_agents):
            self.elements[i].evolve(parameters=parameters, reference_element=reference_particle.elements[i])
        self.updatePersonalBestElements(flock=flock, candidate_elements=self.elements)

    def updatePersonalBestElements(self, flock, candidate_elements):
        candidate_objective_value = computeObjectiveForParticle(flock=flock, particle=self)
        if candidate_objective_value < self.global_best_objective_value:
            self.personal_best_elements = copy.deepcopy(candidate_elements)
            self.global_best_objective_value = candidate_objective_value

class Swarm:
    def __init__(self, parameters, flock):
        self.particles = [Particle(parameters=parameters, flock=flock) for i in range(parameters.number_of_particles)]

        p = min(self.particles, key=lambda p: computeObjectiveForParticle(flock=flock, particle=p))
        self.global_best_particle = copy.deepcopy(p)
        self.global_best_objective_value = computeObjectiveForParticle(flock=flock, particle=self.global_best_particle)

    def evolveParticles(self, parameters, flock):
        for particle in self.particles:
            particle.evolve(parameters=parameters, flock=flock, reference_particle=self.global_best_particle)
            self.updateGlobalBestParticle(flock=flock, candidate_particle=particle)

    def updateGlobalBestParticle(self, flock, candidate_particle):
        candidate_objective_value = computeObjectiveForParticle(flock=flock, particle=candidate_particle)
        if candidate_objective_value < self.global_best_objective_value:
            self.global_best_particle = copy.deepcopy(candidate_particle)
            self.global_best_objective_value = candidate_objective_value


#compute the objective value which would be obtained if the flock moved one time step with respect to the particle
def computeObjectiveForParticle(flock, particle):
    assert(1==2)
    return None
#todo


class Bird:
    def __init__(self, x, y, dx, dy, ddx, ddy):
        self.x_position = x
        self.y_position = y

        self.x_velocity = dx
        self.y_velocity = dy

        self.x_acceleration = ddx
        self.y_acceleration = ddy

def randomBird(parameters):
    return Bird(x=np.random.uniform(0, parameters.bounding_box),
                y=np.random.uniform(0, parameters.bounding_box),
                dx=np.random.uniform(0.25, 0.75),
                dy=np.random.uniform(0.25, 0.75),
                ddx=0.0,
                ddy=0.0)

class Flock:
    def __init__(self, parameters):
        self.birds = [randomBird(parameters=parameters) for i in range(parameters.number_of_agents)]

def generateRandomFlock(parameters):
    return Flock(parameters=parameters)

def temp(parameters):
    current_flock = generateRandomFlock(parameters=parameters)
    best_particles = []
    for time_i in range(parameters.number_of_time_steps):
        best_particle = computeBestParticle(flock=current_flock, parameters=parameters)
        updatePositions(current_flock, best_particle)
        best_particles.append(best_particle)

#computes the best particle for the current position of the flock
#uses particle swarm optimisation
def computeBestParticle(flock, parameters):
    swarm = Swarm(parameters=parameters) #for now, the swarm is randomly generated
    for i in range(parameters.number_of_swarm_steps):
        swarm.evolveParticles(parameters=parameters, flock=flock) #note that this might modify the best particle of the swarm
    return swarm.best_particle