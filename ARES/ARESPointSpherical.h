#ifndef ARES_POINT_SPHERICAL_H_
#define ARES_POINT_SPHERICAL_H_

class ARESPointSpherical{
public:

	double magnitude_;
	double angle_;

	ARESPointSpherical(double magnitude, double angle);

	static ARESPointSpherical createRandomPoint(double lower_bound, double upper_bound);
};

#endif // !ARES_ACCELERATION
