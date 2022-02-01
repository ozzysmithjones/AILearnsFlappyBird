#pragma once

constexpr auto INPUT_LAYER_SIZE = 4;
constexpr auto HIDDEN_LAYER_SIZE = 8;
constexpr auto OUTPUT_LAYER_SIZE = 4;

constexpr auto NUM_HIDDEN_LAYERS = 2;
constexpr auto NUM_LAYERS = NUM_HIDDEN_LAYERS + 2;

constexpr auto INPUT_LAYER = 0;
constexpr auto HIDDEN_LAYER = 1;
constexpr auto OUTPUT_LAYER = NUM_LAYERS - 1;

class NeuralNetwork
{
private:

	unsigned int layerSizes		  [NUM_LAYERS];
	float* valuesByLayer  [NUM_LAYERS];
	float* weightsByLayer [NUM_LAYERS-1];
	float* offsetsByLayer [NUM_LAYERS];

	NeuralNetwork();
	~NeuralNetwork();

	float Sigmoid(float value);
	void SetInputNeuronValue(unsigned int neuronIndex, float value);
	float GetOutputNeuronValue(unsigned int neuronIndex);
	void Process();
	void OptimisedProcess();


};

