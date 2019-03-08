import numpy as np
import copy

class Parameters:
    def __init__(self):
        self.delta_t = 1.0

        #flock parameters
        self.number_of_agents = 5
        self.bounding_box = 3.0
        self.number_of_time_steps = 10
        self.random_upper_bound = 2*np.pi
        self.random_lower_bound = 0.0

        #swarm parameters
        self.cognitive_coefficient = 0.5
        self.social_coefficient = 0.5
        self.inertia_max = 1.0
        self.inertia_min = 0.1
        self.inertia = self.inertia_min
        self.number_of_swarm_steps = 10
        self.number_of_particles = 10

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
        self.personal_best_objective_value = computeObjectiveForParticle(parameters=parameters, flock=flock, particle=self)


    def evolve(self, parameters, flock, reference_particle):
        assert (parameters.number_of_agents == len(self.elements))

        for i in range(parameters.number_of_agents):
            self.elements[i].evolve(parameters=parameters, reference_element=reference_particle.elements[i])
        self.updatePersonalBestElements(flock=flock, candidate_elements=self.elements)

    def updatePersonalBestElements(self, flock, candidate_elements):
        candidate_objective_value = computeObjectiveForParticle(parameters=parameters, flock=flock, particle=self)
        if candidate_objective_value < self.personal_best_objective_value:
            self.personal_best_elements = copy.deepcopy(candidate_elements)
            self.personal_best_objective_value = candidate_objective_value

    def getAcceleration(self, agent_i):
        return self.elements[agent_i].acceleration

class Swarm:
    def __init__(self, parameters, flock):
        self.particles = [Particle(parameters=parameters, flock=flock) for i in range(parameters.number_of_particles)]

        p = min(self.particles, key=lambda p: computeObjectiveForParticle(parameters=parameters, flock=flock, particle=p))
        self.global_best_particle = copy.deepcopy(p)
        self.global_best_objective_value = computeObjectiveForParticle(parameters=parameters, flock=flock, particle=self.global_best_particle)

        self.inertia_counter = 0

    def evolveParticles(self, parameters, flock):
        for particle in self.particles:
            particle.evolve(parameters=parameters, flock=flock, reference_particle=self.global_best_particle)
            self.updateGlobalInformation(parameters=parameters, flock=flock, candidate_particle=particle)

    def updateGlobalInformation(self, parameters, flock, candidate_particle):
        #update global best partical
        candidate_objective_value = computeObjectiveForParticle(parameters=parameters, flock=flock, particle=candidate_particle)
        if candidate_objective_value < self.global_best_objective_value:
            self.global_best_particle = copy.deepcopy(candidate_particle)
            self.global_best_objective_value = candidate_objective_value
            self.inertia_counter = max(self.inertia_counter - 1, 0)
        else:
            self.inertia_counter += 1

        # update inertia
        if self.inertia_counter < 2:
            self.inertia_counter = max(parameters.inertia_min, min(parameters.inertia_max, 2.0 * parameters.inertia))
        else:
            if self.inertia_counter > 5:
                parameters.inertia = max(parameters.inertia_min, min(parameters.inertia_max, 0.5 * parameters.inertia))


def v_matching(vx, vy, num):
    assert(len(vx) > 0)
    assert(len(vy) > 0)

    sum = 0.0
    diff = 0.0
    for i in range(0, num):
        for j in range(i+1, num):
            diff = np.linalg.norm([vx[i] - vx[j], vy[i] - vy[j]]) / (np.linalg.norm([vx[i], vy[i]]) + np.linalg.norm([vx[j], vy[j]]))
            sum += diff * diff
    return sum

def computeObjective(parameters, flock):
    benefit = 0
    la = 0.5 - np.pi / 8.0
    w = 1.0
    d0 = 1.0
    u_sigma1 = 5.0
    u_sigma2 = 5.0
    d_sigma1 = 1.0 / 0.3
    d_sigma2 = 1.0 / 0.7
    heading = np.pi/2
    headFit = 0

    x = np.array([[bird.x_position, bird.y_position] for bird in flock.birds])
    dx = np.array([[bird.x_velocity, bird.y_velocity] for bird in flock.birds])
    obstacle = 0.0

    for i in range(len(flock.birds)):

        #x = [bird.x_position, bird.y_position] #x = agent.model.X
        #dx = [bird.x_velocity, bird.y_velocity] #dx = agent.model.dX
        A = dx[i,0]
        B = dx[i,1]
        C = -dx[i,1] * x[i,1] - dx[i,0] * x[i,0]
        ub_j = 0

        # heading direction
        if (np.math.atan2(dx[i,0], dx[i,1]) != heading):
            headFit += abs(np.math.atan2(dx[i,0], dx[i,1]) - heading)

        for  j in range(len(flock.birds)):
            if i != j and np.linalg.norm(x[j,:]) < np.linalg.norm(x[i,:]):

                if (dx[i,0] == 0.0):
                    px = x[j,0]
                    py = x[i,1]
                else:
                    if (dx[i,1] == 0.0):
                        px = x[i,0]
                        py = x[j,1]
                    else:
                        k = -dx[i,0] / dx[i,1]
                        px = (k * x[i,0] + x[j,0] / k + x[j,1] - x[i,1]) / (k + 1.0 / k)
                        py = -1.0 / k * (px - x[j,0]) + x[j,1]

                side = A * x[j,0] + B * x[j,1] + C
                h_dis = np.linalg.norm([px - x[i,0], py - x[i,1]])
                v_dis = abs(side) / np.linalg.norm([A, B])

                sm = np.math.erf((h_dis - (w - la)) * np.sqrt(2.0) * 8.0)
                dot_prod = (dx[i,0] * dx[j,0] + dx[i,1] * dx[j,1]) / (np.linalg.norm([dx[i,0], dx[i,1]]) * np.linalg.norm([dx[j,0], dx[j,1]]))

                from scipy.stats import multivariate_normal
                if (side > 0.0 and h_dis >= w - la):
                    ub_j += dot_prod * sm * multivariate_normal.pdf([h_dis - (2.0 * w - la), v_dis - d0], [0,0], [u_sigma1, u_sigma2])
                else:
                    if (side >= 0.0 and h_dis < w - la):
                        ub_j += sm * multivariate_normal.pdf([h_dis, v_dis], [0,0], [d_sigma1, d_sigma2])

                angle = np.pi / 6.0

                blocks = [[0.0, 0.0] for i in range(parameters.number_of_agents)]

                if (side >= 0.0 and (h_dis < w or (h_dis - w) / v_dis < np.tan(angle))):
                  blocks[j][0] = np.arctan(v_dis / (h_dis + w))
                  blocks[j][1] = np.arctan2(v_dis, h_dis - w)
                  if (blocks[j][0] < np.pi / 2.0 - angle):
                    blocks[j][0] = np.pi / 2.0 - angle
                  if (blocks[j][1] > np.pi / 2.0 + angle):
                    blocks[j][1] = np.pi / 2.0 + angle
                  obstacle += (blocks[j][1] - blocks[j][0]) / (angle)





        if (ub_j < 1.0):
            benefit += ub_j
        else:
            benefit += 1.0


    fit = pow(len(flock.birds) - 1.0 - benefit, 2) + v_matching(vx=dx[:,0], vy=dx[:,1], num=parameters.number_of_agents) + pow(obstacle, 2)# + pow(headFit/len(agents), 2)
    return fit


#compute the objective value which would be obtained if the flock moved one time step with respect to the particle
def computeObjectiveForParticle(parameters, flock, particle):
    initial_flock = copy.deepcopy(flock)
    initial_flock.updatePositions(parameters=parameters, particle=particle)
    objective_value = computeObjective(parameters=parameters, flock=initial_flock)
    return objective_value

class Bird:
    def __init__(self, x, y, dx, dy, ddx, ddy):
        self.x_position = x
        self.y_position = y

        self.x_velocity = dx
        self.y_velocity = dy

        self.x_acceleration = ddx
        self.y_acceleration = ddy

    #note that acceleration might change due to trimming :o
    def move(self, parameters, acceleration):
        # trim - reduce the acceleration if it is too big
        norm = np.linalg.norm([self.x_velocity, self.y_velocity])
        if acceleration.magnitude >= norm:
            acceleration.magnitude = norm

        #convert the received acceleration to x y coordinates
        self.x_acceleration = acceleration.magnitude * np.cos(acceleration.angle)
        self.y_acceleration = acceleration.magnitude * np.sin(acceleration.angle)

        #update velocities
        self.x_velocity += (self.x_acceleration * parameters.delta_t)
        self.y_velocity += (self.y_acceleration * parameters.delta_t)

        #update positions
        self.x_position += (self.x_velocity * parameters.delta_t)
        self.y_position += (self.y_velocity * parameters.delta_t)

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

    def updatePositions(self, parameters, particle):
        assert(len(particle.elements) == parameters.number_of_agents)

        for agent_i in range(parameters.number_of_agents):
            self.birds[agent_i].move(parameters=parameters, acceleration=particle.getAcceleration(agent_i))

def generateRandomFlock(parameters):
    return Flock(parameters=parameters)

def tempName(parameters):
    current_flock = generateRandomFlock(parameters=parameters)
    current_objective_value = computeObjective(parameters=parameters, flock=current_flock)
    best_particles = []

    for time_i in range(parameters.number_of_time_steps):
        #todo properly include the horizon approach and levels
        print(time_i)
        print("current objective value: {}".format(current_objective_value))
        (best_particle, best_objective_value) = computeBestParticle(flock=current_flock, parameters=parameters)
        hehe = 0
        while best_objective_value > current_objective_value:
            (best_particle, best_objective_value) = computeBestParticle(flock=current_flock, parameters=parameters)
            hehe += 1

            if hehe > 10:
                print("Sadness!")
                break

        current_flock.updatePositions(parameters=parameters, particle=best_particle)
        current_objective_value = best_objective_value
        best_particles.append(best_particle)

        #TODO FIX FOR THE BIRDS LOOOOOOOOOOOOOOOOOOL
        #parameters.random_upper_bound
        #options[b].ub[j] = delta * norm(info[b].cvx[k], info[b].cvy[k]);
        #printf("options_ub: %f\t, ind: %d\t, b: %d\n", options[b].ub[j], k, b);
        #j++;













        import matplotlib.pyplot as plt

        X = np.array([[bird.x_position, bird.y_position] for bird in current_flock.birds])
        dX = np.array([[bird.x_velocity, bird.y_velocity] for bird in current_flock.birds])

        Num = parameters.number_of_agents
        fig, ax = plt.subplots()
        ax.plot(X[:, 0], X[:, 1], 'ro')
        N = [x for x in range(0, Num)]
        for i, txt in enumerate(N):
            ax.annotate(txt, (X[-i - 1, 0], X[-i - 1, 1]))
        ax.quiver(X[:, 0], X[:, 1], dX[:, 0], dX[:, 1])
        plt.show()

#computes the best particle for the current position of the flock
#uses particle swarm optimisation
def computeBestParticle(flock, parameters):
    swarm = Swarm(parameters=parameters, flock=flock) #for now, the swarm is randomly generated
    for i in range(parameters.number_of_swarm_steps):
        swarm.evolveParticles(parameters=parameters, flock=flock) #note that this might modify the best particle of the swarm
    return (swarm.global_best_particle, swarm.global_best_objective_value)

parameters = Parameters()
tempName(parameters=parameters)