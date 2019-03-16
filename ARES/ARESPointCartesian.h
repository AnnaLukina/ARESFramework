#ifndef ARES_POINT_CARTESIAN_H_
#define ARES_POINT_CARTESIAN_H_

#include <math.h>

#include "ARESPointSpherical.h"

class ARESPointCartesian
{
public:
	double x_, y_;

	ARESPointCartesian(double x, double y);
	ARESPointCartesian& operator=(ARESPointSpherical &p);
	double norm() const;
};

#endif // !ARES_POINT_H
