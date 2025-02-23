#include "Timer.h"
using namespace std::chrono;

Timer::Timer()
	:
	last(steady_clock::now())
{
}

float Timer::Mark()
{
	const auto old = last;
	last = steady_clock::now();
	const duration<float> time = last - old;
	return time.count();
}
