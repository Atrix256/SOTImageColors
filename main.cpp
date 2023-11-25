#define _CRT_SECURE_NO_WARNINGS // for stb

// Settings
#define DETERMINISTIC() true
static const int c_numIterations = 100;
static const int c_batchSize = 16;

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <random>
#include <vector>
#include <direct.h>
#include <stdio.h>

using Vec3 = float[3];

struct ImageData
{
	int width = 0;
	int height = 0;
	std::vector<float> pixels;
};

struct Settings
{
	const char* srcImageFileName = nullptr;
	ImageData srcimage;

	const char* targetImageFileName = nullptr;
	ImageData targetImage;
	float weight = 1.0f;

	std::vector<float> result;
};

inline float Lerp(float A, float B, float t)
{
	return A * (1.0f - t) + B * t;
}

std::mt19937 GetRNG(int index)
{
	#if DETERMINISTIC()
	std::mt19937 ret(index);
	#else
	std::random_device rd;
	std::mt19937 ret(rd());
	#endif
	return ret;
}

bool LoadImageAsFloat(const char* fileName, ImageData& imageData)
{
	int c;
	stbi_uc* pixelsU8 = stbi_load(fileName, &imageData.width, &imageData.height, &c, 3);
	if (!pixelsU8)
		return false;

	imageData.pixels.resize(imageData.width * imageData.height * 3);

	for (size_t i = 0; i < imageData.width * imageData.height * 3; ++i)
		imageData.pixels[i] = float(pixelsU8[i]);

	stbi_image_free(pixelsU8);
	return true;
}

bool SaveFloatImage(const ImageData& imageData, const char* fileName)
{
	std::vector<unsigned char> pixels(imageData.width * imageData.height * 3);
	for (size_t index = 0; index < pixels.size(); ++index)
		pixels[index] = (unsigned char)std::max(std::min(imageData.pixels[index], 255.0f), 0.0f);

	return stbi_write_png(fileName, imageData.width, imageData.height, 3, pixels.data(), 0) == 1;
}

void SlicedOptimalTransport(Settings& settings)
{
	printf("\n%s\n\n", settings.targetImageFileName);

	const uint32_t c_numPixels = settings.srcimage.width * settings.srcimage.height;

	std::vector<float>& current = settings.result;
	current = settings.srcimage.pixels;

	// Per batch data
	// Each batch has it's own data so the batches can be parallelized
	struct BatchData
	{
		BatchData(uint32_t numPixels)
		{
			currentSorted.resize(numPixels);
			targetSorted.resize(numPixels);
			for (uint32_t i = 0; i < numPixels; ++i)
			{
				currentSorted[i] = i;
				targetSorted[i] = i;
			}

			currentProjections.resize(numPixels);
			targetProjections.resize(numPixels);

			batchDirections.resize(numPixels * 3);
		}

		std::vector<uint32_t> currentSorted;
		std::vector<uint32_t> targetSorted;

		std::vector<float> currentProjections;
		std::vector<float> targetProjections;

		std::vector<float> batchDirections;
	};
	std::vector<BatchData> allBatchData(c_batchSize, BatchData(c_numPixels));

	// For each iteration
	for (int iteration = 0; iteration < c_numIterations; ++iteration)
	{
		// Do the batches in parallel
		#pragma omp parallel for
		for (int batchIndex = 0; batchIndex < c_batchSize; ++batchIndex)
		{
			BatchData& batchData = allBatchData[batchIndex];

			std::mt19937 rng = GetRNG(iteration * c_batchSize + batchIndex);
			std::normal_distribution<float> normalDist(0.0f, 1.0f);

			// Make a uniform random unit vector by generating 3 normal distributed values and normalizing the result.
			Vec3 direction;
			direction[0] = normalDist(rng);
			direction[1] = normalDist(rng);
			direction[2] = normalDist(rng);
			float length = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
			direction[0] /= length;
			direction[1] /= length;
			direction[2] /= length;

			// project current and target
			for (size_t i = 0; i < c_numPixels; ++i)
			{
				batchData.currentProjections[i] =
					direction[0] * current[i * 3 + 0] +
					direction[1] * current[i * 3 + 1] +
					direction[2] * current[i * 3 + 2];

				batchData.targetProjections[i] =
					direction[0] * settings.targetImage.pixels[i * 3 + 0] +
					direction[1] * settings.targetImage.pixels[i * 3 + 1] +
					direction[2] * settings.targetImage.pixels[i * 3 + 2];
			}

			// sort current and target
			std::sort(batchData.currentSorted.begin(), batchData.currentSorted.end(),
				[&] (uint32_t a, uint32_t b)
				{
					return batchData.currentProjections[a] < batchData.currentProjections[b];
				}
			);

			std::sort(batchData.targetSorted.begin(), batchData.targetSorted.end(),
				[&](uint32_t a, uint32_t b)
				{
					return batchData.targetProjections[a] < batchData.targetProjections[b];
				}
			);

			// update batchDirections
			for (size_t i = 0; i < c_numPixels; ++i)
			{
				float projDiff = batchData.targetProjections[batchData.targetSorted[i]] - batchData.currentProjections[batchData.currentSorted[i]];

				batchData.batchDirections[batchData.currentSorted[i] * 3 + 0] = direction[0] * projDiff;
				batchData.batchDirections[batchData.currentSorted[i] * 3 + 1] = direction[1] * projDiff;
				batchData.batchDirections[batchData.currentSorted[i] * 3 + 2] = direction[2] * projDiff;
			}
		}

		// average all batch directions into batchDirections[0]
		{
			for (int batchIndex = 1; batchIndex < c_batchSize; ++batchIndex)
			{
				float alpha = 1.0f / float(batchIndex + 1);
				for (size_t i = 0; i < c_numPixels * 3; ++i)
					allBatchData[0].batchDirections[i] = Lerp(allBatchData[0].batchDirections[i], allBatchData[batchIndex].batchDirections[i], alpha);
			}
		}

		// update current
		float totalDistance = 0.0f;
		for (size_t i = 0; i < c_numPixels; ++i)
		{
			float adjust[3] = {
				allBatchData[0].batchDirections[i * 3 + 0],
				allBatchData[0].batchDirections[i * 3 + 1],
				allBatchData[0].batchDirections[i * 3 + 2]
			};

			current[i * 3 + 0] += adjust[0];
			current[i * 3 + 1] += adjust[1];
			current[i * 3 + 2] += adjust[2];

			totalDistance += std::sqrt(adjust[0] * adjust[0] + adjust[1] * adjust[1] + adjust[2] * adjust[2]);
		}

		printf("[%i] %f\n", iteration, totalDistance / float(c_numPixels));
	}
}

void InterpolateColorHistogram(Settings& settings, const char* outputFileNameBase)
{
	char outputFileName[1024];
	sprintf_s(outputFileName, "%s.png", outputFileNameBase);

	char outputFileNameCSV[1024];
	sprintf_s(outputFileNameCSV, "%s.csv", outputFileNameBase);

	printf("==================================\n%s\n==================================\n", outputFileName);

	// load up the source image
	LoadImageAsFloat(settings.srcImageFileName, settings.srcimage);

	// load the target image and verify it's compatible
	LoadImageAsFloat(settings.targetImageFileName, settings.targetImage);
	if (settings.targetImage.width != settings.srcimage.width || settings.targetImage.height != settings.srcimage.height)
	{
		printf("ERROR: image %s is %ix%i, but should be %ix%i like %s.\n",
			settings.targetImageFileName, settings.targetImage.width, settings.targetImage.height,
			settings.srcimage.width, settings.srcimage.height, settings.srcImageFileName);
		return;
	}

	// do optimal transport to get the per pixel delta to get from the source image to the target image
	SlicedOptimalTransport(settings);

	// Do interpolation
	ImageData output = settings.srcimage;
	for (size_t valueIndex = 0; valueIndex < output.width * output.height * 3; ++valueIndex)
		output.pixels[valueIndex] = Lerp(settings.srcimage.pixels[valueIndex], settings.result[valueIndex], settings.weight);

	// Save output image
	SaveFloatImage(output, outputFileName);
}

int main(int argc, char** argv)
{
	_mkdir("out");

	{
		Settings settings;
		settings.srcImageFileName = "images/florida.png";
		settings.targetImageFileName = "images/dunes.png";
		InterpolateColorHistogram(settings, "out/florida-dunes");
	}

	{
		Settings settings;
		settings.srcImageFileName = "images/florida.png";
		settings.targetImageFileName = "images/turtle.png";
		InterpolateColorHistogram(settings, "out/florida-turtle");
	}

	{
		Settings settings;
		settings.srcImageFileName = "images/florida.png";
		settings.targetImageFileName = "images/bigcat.png";
		InterpolateColorHistogram(settings, "out/florida-bigcat");
	}

	return 0;
}

/*
TODO:
* get rid of settings struct, make this more ad hoc
* make a csv of total movement over time and graph it to see if it's enough steps
* output how long it took. not super important though
* clean up code?
* make an interpolation example with 2 images
* also with 3.
! mention in the post that you could do barycentric interpolation with multiple images. and could even go "outside of the simplex" with those coordinates to move away from histograms etc.
! mention that the sorting is the slowest part, per the profiler. same as the other post. could multithread it but :shrug:
! say how long it took, and what resolution image
* mention how you can do all the batches in parallel
*/
