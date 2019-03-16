#ifndef ARES_LIBRARY_H_
#define ARES_LIBRARY_H_

#include <assert.h>
#include <math.h>

#include "ARESPointCartesian.h"

class ARESLibrary 
{
public:
	static long double PI() 
	{ 
		return 3.141592653589793238462643383279502884L; 
	}

	static double random_uniform(double lower_bound, double upper_bound)
	{
		assert(1 == 2);
		return -1;
	}
	
};

#endif // !ARES_LIBRARY
