#include <cfloat>
#include <limits>
#include "Random.h"


Random::Random(uint64_t seed)
	: rng(dev()), distribution(-1.0, std::nextafter(1.0f, std::numeric_limits<float>::max()))
{
	std::seed_seq ss{uint32_t(seed & 0xffffffff), uint32_t(seed >> 32)};
	rng.seed();
}

float Random::Value()
{
	return distribution(rng);
}
