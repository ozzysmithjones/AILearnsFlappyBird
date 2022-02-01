
#include <math.h>
#include <immintrin.h>
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork()
{
	layerSizes[INPUT_LAYER] = HIDDEN_LAYER_SIZE;
	layerSizes[OUTPUT_LAYER] = OUTPUT_LAYER_SIZE;
	for (unsigned int i = HIDDEN_LAYER; i < (HIDDEN_LAYER + NUM_HIDDEN_LAYERS); i++)
	{
		layerSizes[i] = HIDDEN_LAYER_SIZE;
	}

	valuesByLayer[INPUT_LAYER] = new float[INPUT_LAYER_SIZE] {0};
	valuesByLayer[OUTPUT_LAYER] = new float[OUTPUT_LAYER_SIZE] {0};
	for (unsigned int i = HIDDEN_LAYER; i < (HIDDEN_LAYER + NUM_HIDDEN_LAYERS); i++)
	{
		valuesByLayer[i] = new float[HIDDEN_LAYER_SIZE] {0};
	}

	weightsByLayer[INPUT_LAYER] = new float[HIDDEN_LAYER_SIZE] {0};
	for (unsigned int i = HIDDEN_LAYER; i < (HIDDEN_LAYER + NUM_HIDDEN_LAYERS)-1; i++)
	{
		weightsByLayer[i] = new float[HIDDEN_LAYER_SIZE] {0};
	}
	weightsByLayer[(HIDDEN_LAYER + NUM_HIDDEN_LAYERS) - 1] = new float[OUTPUT_LAYER_SIZE];

	offsetsByLayer[INPUT_LAYER] = nullptr;
	offsetsByLayer[OUTPUT_LAYER] = new float[OUTPUT_LAYER_SIZE] {0};
	for (unsigned int i = HIDDEN_LAYER; i < (HIDDEN_LAYER + NUM_HIDDEN_LAYERS); i++)
	{
		offsetsByLayer[i] = new float[HIDDEN_LAYER_SIZE] {0};
	}
}

NeuralNetwork::~NeuralNetwork()
{
	for (unsigned int i = 0; i < NUM_LAYERS; i++)
	{
		delete[] valuesByLayer[i];
	}

	for (unsigned int i = 1; i < NUM_LAYERS; i++)
	{
		delete[] offsetsByLayer[i];
	}

	for (unsigned int i = 0; i < NUM_LAYERS-1; i++)
	{
		delete[] weightsByLayer[i];
	}
}

float NeuralNetwork::Sigmoid(float value)
{
	return 1.0f / (1 + exp(-value));
}

void NeuralNetwork::SetInputNeuronValue(unsigned int neuronIndex, float value)
{
	valuesByLayer[INPUT_LAYER][neuronIndex] = value;
}

float NeuralNetwork::GetOutputNeuronValue(unsigned int neuronIndex)
{
	return valuesByLayer[OUTPUT_LAYER][neuronIndex];
}

void NeuralNetwork::Process()
{
	//Set all node values past the Input to zero

	for (unsigned int i = HIDDEN_LAYER; i < NUM_LAYERS; i++)
	{
		unsigned int layerSize = layerSizes[i];

		for (unsigned int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
		{
			valuesByLayer[i][nodeIndex] = 0;
		}
	}

	//Forward process all of the neurons.

	for (unsigned int i = INPUT_LAYER; i < OUTPUT_LAYER; i++)
	{
		const unsigned int layerSize = layerSizes[i];
		const unsigned int nextLayerSize = layerSizes[i + 1];

		for (unsigned int nodeIndex = 0; nodeIndex < layerSize; i++)
		{
			for (unsigned int nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++)
			{
				valuesByLayer[i + 1][nextNodeIndex] += valuesByLayer[i][nodeIndex] * weightsByLayer[i][nextNodeIndex];
			}
		}

		for (unsigned int nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++)
		{
			float& val = valuesByLayer[i + 1][nextNodeIndex];
			val = Sigmoid(offsetsByLayer[i + 1][nextNodeIndex] + val);
		}
	}
}

void NeuralNetwork::OptimisedProcess()
{
	static_assert((INPUT_LAYER_SIZE % 4) == 0,  "Input  layer size must be multiple of 4 for vector intrinsics");
	static_assert((HIDDEN_LAYER_SIZE % 4) == 0, "Hidden layer size must be multiple of 4 for vector intrinsics");
	static_assert((OUTPUT_LAYER_SIZE % 4) == 0, "Output layer size must be multiple of 4 for vector intrinsics");

	const __m128 zero = _mm_setzero_ps();

	for (unsigned int i = HIDDEN_LAYER; i < NUM_LAYERS; i++)
	{
		const unsigned int layerSize = layerSizes[i];
		for (unsigned int nodeIndex = 0; nodeIndex < layerSize; nodeIndex += 4)
		{
			_mm_storeu_ps(valuesByLayer[i] + nodeIndex, zero);
		}
	}

	for (unsigned int i = INPUT_LAYER; i < OUTPUT_LAYER; i++)
	{
		const unsigned int layerSize = layerSizes[i];
		const unsigned int nextLayerSize = layerSizes[i + 1];

		for (unsigned int nodeIndex = 0; nodeIndex < layerSize; i++)
		{
			const __m128 value = _mm_set1_ps(valuesByLayer[i][nodeIndex]);

			for (unsigned int nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex += 4)
			{
				float* next = valuesByLayer[i + 1] + nextNodeIndex;
				const __m128 nextValues = _mm_loadu_ps(next);
				const __m128 nextWeights = _mm_loadu_ps(weightsByLayer[i] + nextNodeIndex);
				_mm_storeu_ps(next, _mm_add_ps(nextValues, _mm_mul_ps(value,nextWeights)));
			}
		}

		for (unsigned int nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex += 4)
		{
			float* next = valuesByLayer[i + 1] + nextNodeIndex;
			const __m128 nextValues = _mm_loadu_ps(next);
			const __m128 nextOffsets = _mm_loadu_ps(offsetsByLayer[i + 1] + nextNodeIndex);
			_mm_storeu_ps(next, _mm_add_ps(nextValues, nextOffsets));
		}

		//sigmoid
		for (unsigned int nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++)
		{
			float& val = valuesByLayer[i + 1][nextNodeIndex];
			val = Sigmoid(val);
		}
	}
}
