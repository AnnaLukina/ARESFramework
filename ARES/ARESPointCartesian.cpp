#include "ARESPointCartesian.h"

ARESPointCartesian::ARESPointCartesian(double x, double y) :x_(x), y_(y) {}

ARESPointCartesian& ARESPointCartesian::operator=(ARESPointSpherical &p)
{
	x_ = p.magnitude_ * cos(p.angle_);
	y_ = p.magnitude_ * sin(p.angle_);
	return *this;
}

double ARESPointCartesian::norm() const
{
	return sqrt(x_*x_ + y_ * y_);
}