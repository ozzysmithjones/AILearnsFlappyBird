#include <math.h>
#include <exception>
#include <stdlib.h>
#include <immintrin.h>
#include <algorithm>
#include "NeuralNetwork.h"
#include "Random.h"

static Random random(235);

void NeuralNetwork::AllocateLayers()
{
	layerSizes[INPUT_LAYER] = INPUT_LAYER_SIZE;
	layerSizes[OUTPUT_LAYER] = OUTPUT_LAYER_SIZE;
	for (unsigned int i = HIDDEN_LAYER; i < (HIDDEN_LAYER + NUM_HIDDEN_LAYERS); i++)
	{
		layerSizes[i] = HIDDEN_LAYER_SIZE;
	}

	valuesByLayer[INPUT_LAYER] = AllocArray(INPUT_LAYER_SIZE);
	valuesByLayer[OUTPUT_LAYER] = AllocArray(OUTPUT_LAYER_SIZE);
	for (unsigned int i = HIDDEN_LAYER; i < (HIDDEN_LAYER + NUM_HIDDEN_LAYERS); i++)
	{
		valuesByLayer[i] = AllocArray(HIDDEN_LAYER_SIZE);
	}

	weightsByLayer[INPUT_LAYER] = AllocArray(HIDDEN_LAYER_SIZE);
	weightsByLayer[(HIDDEN_LAYER + NUM_HIDDEN_LAYERS) - 1] = AllocArray(OUTPUT_LAYER_SIZE);
	for (unsigned int i = HIDDEN_LAYER; i < (HIDDEN_LAYER + NUM_HIDDEN_LAYERS) - 1; i++)
	{
		weightsByLayer[i] = AllocArray(HIDDEN_LAYER_SIZE);
	}

	offsetsByLayer[INPUT_LAYER] = AllocArray(HIDDEN_LAYER_SIZE);
	offsetsByLayer[(HIDDEN_LAYER + NUM_HIDDEN_LAYERS) - 1] = AllocArray(OUTPUT_LAYER_SIZE);
	for (unsigned int i = HIDDEN_LAYER; i < (HIDDEN_LAYER + NUM_HIDDEN_LAYERS) - 1; i++)
	{
		offsetsByLayer[i] = AllocArray(HIDDEN_LAYER_SIZE);
	}
}

NeuralNetwork::NeuralNetwork()
{
	AllocateLayers();

	//Randomize the initial weights and offsets.

	const unsigned int layerSize = layerSizes[INPUT_LAYER];
	for (unsigned int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
	{
		valuesByLayer[INPUT_LAYER][nodeIndex] = 0;
	}

	for (unsigned int i = INPUT_LAYER; i < OUTPUT_LAYER; i++)
	{
		const unsigned int nextLayerSize = layerSizes[i + 1];

		for (unsigned int nodeIndex = 0; nodeIndex < nextLayerSize; nodeIndex++)
		{
			weightsByLayer[i][nodeIndex] = random.Value();
			offsetsByLayer[i][nodeIndex] = random.Value();
		}
	}
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& other)
{
	AllocateLayers();

	for (unsigned int i = INPUT_LAYER; i < OUTPUT_LAYER; i++)
	{
		memcpy(valuesByLayer[i], other.valuesByLayer[i], layerSizes[i] * sizeof(float));
		memcpy(weightsByLayer[i], other.weightsByLayer[i], layerSizes[i + 1] * sizeof(float));
		memcpy(offsetsByLayer[i], other.offsetsByLayer[i], layerSizes[i + 1] * sizeof(float));
	}

	memcpy(valuesByLayer[OUTPUT_LAYER], other.valuesByLayer[OUTPUT_LAYER], layerSizes[OUTPUT_LAYER] * sizeof(float));
}

NeuralNetwork::~NeuralNetwork()
{
	for (unsigned int i = 0; i < NUM_LAYERS; i++)
	{
		FreeArray(valuesByLayer[i]);
	}

	for (unsigned int i = 1; i < NUM_LAYERS - 1; i++)
	{
		FreeArray(offsetsByLayer[i]);
		FreeArray(weightsByLayer[i]);
	}
}

float* NeuralNetwork::AllocArray(const size_t numElements)
{
	const auto arr = (float*)_aligned_malloc(numElements * sizeof(float), 16);

	if (arr == nullptr)
	{
		throw std::bad_alloc();
	}

	return arr;
}

void NeuralNetwork::FreeArray(float* arr)
{
	_aligned_free(arr);
}

float NeuralNetwork::Sigmoid(const float value)
{
	return 1.0f / (1 + expf(-value));
}

void NeuralNetwork::Mutate() const
{
	const auto layer = random.Range<unsigned int>(0, NUM_LAYERS - 1);
	const auto nodeIndex = random.Range<unsigned int>(0, layerSizes[layer]);

	if (random.Value() > 0.0f)
	{
		weightsByLayer[layer][nodeIndex] = random.Value();
	}
	else
	{
		offsetsByLayer[layer][nodeIndex] = random.Value();
	}
}

void NeuralNetwork::SetInputNeuronValue(unsigned int neuronIndex, float value) const
{
	valuesByLayer[INPUT_LAYER][neuronIndex] = value;
}

float NeuralNetwork::GetOutputNeuronValue(unsigned int neuronIndex) const
{
	return valuesByLayer[OUTPUT_LAYER][neuronIndex];
}

void NeuralNetwork::Process() const
{
	//Set all node values past the Input to zero

	for (unsigned int i = HIDDEN_LAYER; i < NUM_LAYERS; i++)
	{
		const unsigned int layerSize = layerSizes[i];

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

		for (unsigned int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
		{
			for (unsigned int nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++)
			{
				valuesByLayer[i + 1][nextNodeIndex] += valuesByLayer[i][nodeIndex] * weightsByLayer[i][nextNodeIndex];
			}
		}

		for (unsigned int nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++)
		{
			float& val = valuesByLayer[i + 1][nextNodeIndex];
			val = Sigmoid(offsetsByLayer[i][nextNodeIndex] + val);
		}
	}
}

void NeuralNetwork::OptimisedProcess() const
{
	static_assert(sizeof(float) == 4, "float must be sizeof(4) for correct alignment in vector intrinsics");
	static_assert((INPUT_LAYER_SIZE % 4) == 0, "Input  layer size must be multiple of 4 for vector intrinsics");
	static_assert((HIDDEN_LAYER_SIZE % 4) == 0, "Hidden layer size must be multiple of 4 for vector intrinsics");
	static_assert((OUTPUT_LAYER_SIZE % 4) == 0, "Output layer size must be multiple of 4 for vector intrinsics");

	const __m128 zero = _mm_setzero_ps();

	for (unsigned int i = HIDDEN_LAYER; i < NUM_LAYERS; i++)
	{
		const unsigned int layerSize = layerSizes[i];
		for (unsigned int nodeIndex = 0; nodeIndex < layerSize; nodeIndex += 4)
		{
			_mm_store_ps(valuesByLayer[i] + nodeIndex, zero);
		}
	}

	for (unsigned int i = INPUT_LAYER; i < OUTPUT_LAYER; i++)
	{
		const unsigned int layerSize = layerSizes[i];
		const unsigned int nextLayerSize = layerSizes[i + 1];

		for (unsigned int nodeIndex = 0; nodeIndex < layerSize; nodeIndex++)
		{
			const __m128 value = _mm_set1_ps(valuesByLayer[i][nodeIndex]);

			for (unsigned int nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex += 4)
			{
				float* next = valuesByLayer[i + 1] + nextNodeIndex;
				const __m128 nextValues = _mm_load_ps(next);
				const __m128 nextWeights = _mm_load_ps(weightsByLayer[i] + nextNodeIndex);
				_mm_store_ps(next, _mm_add_ps(nextValues, _mm_mul_ps(value, nextWeights)));
			}
		}

		for (unsigned int nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex += 4)
		{
			float* next = valuesByLayer[i + 1] + nextNodeIndex;
			const __m128 nextValues = _mm_load_ps(next);
			const __m128 nextOffsets = _mm_load_ps(offsetsByLayer[i] + nextNodeIndex);
			_mm_store_ps(next, _mm_add_ps(nextValues, nextOffsets));
		}

		//sigmoid
		for (unsigned int nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++)
		{
			float& val = valuesByLayer[i + 1][nextNodeIndex];
			val = Sigmoid(val);
		}
	}
}
