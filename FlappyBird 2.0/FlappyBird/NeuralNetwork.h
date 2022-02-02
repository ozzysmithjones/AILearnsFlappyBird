#pragma once


class Random;

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
	unsigned int layerSizes[NUM_LAYERS]{0};
	float* valuesByLayer[NUM_LAYERS]{nullptr};
	float* weightsByLayer[NUM_LAYERS - 1]{nullptr};
	float* offsetsByLayer[NUM_LAYERS - 1]{nullptr};


	void AllocateLayers();
	static float* AllocArray(const size_t numElements);
	static void FreeArray(float* arr);
	static float Sigmoid(const float value);

public:
	NeuralNetwork();
	NeuralNetwork(const NeuralNetwork& other);
	~NeuralNetwork();

	void Mutate() const;
	void SetInputNeuronValue(unsigned int neuronIndex, float value) const;
	float GetOutputNeuronValue(unsigned int neuronIndex) const;
	void Process() const;
	void OptimisedProcess() const;
};
