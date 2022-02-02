#pragma once
#include <array>
#include "NeuralNetwork.h"
#include "AIController.h"


class NeuralNetworkController :
	public AIController
{
private:
	std::array<NeuralNetwork, 200> neuralNetworks;
};
