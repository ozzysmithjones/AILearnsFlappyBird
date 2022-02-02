#pragma once
#include <random>

class Random
{
private:
	//Random number generation,
	std::random_device dev;
	std::mt19937_64 rng;
	const std::uniform_real_distribution<float> distribution;

public:
	Random(uint64_t seed);

	float Value();

	/**
	 * \brief Returns a random value in the range [min, max) (inclusive, exclusive)
	 */
	template <typename T>
	T Range(const T min, const T max)
	{
		const std::uniform_int_distribution<T> dist(min, max);
		return dist(rng);
	}
};
