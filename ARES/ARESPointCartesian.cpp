#include "ARESPointCartesian.h"
#include "ARESLibrary.h"

ARESPointCartesian::ARESPointCartesian(double x, double y) :x_(x), y_(y) {}

ARESPointCartesian& ARESPointCartesian::operator=(const ARESPointSpherical &p)
{
	x_ = p.magnitude_ * cos(p.angle_);
	y_ = p.magnitude_ * sin(p.angle_);
	return *this;
}

double ARESPointCartesian::norm() const
{
	return sqrt(x_*x_ + y_ * y_);
}

ARESPointCartesian ARESPointCartesian::createRandomPoint(double lower_bound, double upper_bound)
{
	return ARESPointCartesian(ARESLibrary::random_uniform(lower_bound, upper_bound), ARESLibrary::random_uniform(lower_bound, upper_bound));
}
