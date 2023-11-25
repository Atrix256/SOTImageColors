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
#include <chrono>

struct ImageData
{
	int width = 0;
	int height = 0;
	std::vector<float> pixels;
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

bool LoadImageAsFloat(ImageData& imageData, const char* fileName)
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

void SlicedOptimalTransport(const ImageData& srcImage, const ImageData& targetImage, std::vector<float>& results, const char* outputFileNameCSV)
{
	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	printf("==================================\nCalculating Optimal Transport - %s\n==================================\n", outputFileNameCSV);

	FILE* file = nullptr;
	fopen_s(&file, outputFileNameCSV, "wb");
	fprintf(file, "\"Iteration\",\"Avg. Movement\"\n");

	const uint32_t c_numPixels = srcImage.width * srcImage.height;

	// start the results at the starting point - the source image
	results = srcImage.pixels;
	std::vector<float>& current = results; // current is an alias of results, to make the code make more sense

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
			float direction[3];
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
					direction[0] * targetImage.pixels[i * 3 + 0] +
					direction[1] * targetImage.pixels[i * 3 + 1] +
					direction[2] * targetImage.pixels[i * 3 + 2];
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
		fprintf(file, "\"%i\",\"%f\"\n", iteration, totalDistance / float(c_numPixels));
	}

	fclose(file);

	float elpasedSeconds = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - start).count();
	printf("\n%0.2f seconds\n\n", elpasedSeconds);	
}

void InterpolateColorHistogram1D(const ImageData& srcImage, const std::vector<float>& target, float weight, const char* outputFileName)
{
	// 1D barycentric coordinates. They add up to 1.0.
	float u = 1.0f - weight;
	float v = weight;

	// Do interpolation
	ImageData output = srcImage;
	for (size_t valueIndex = 0; valueIndex < output.width * output.height * 3; ++valueIndex)
		output.pixels[valueIndex] = srcImage.pixels[valueIndex] * u + target[valueIndex] * v;

	// Save output image
	SaveFloatImage(output, outputFileName);
}

void InterpolateColorHistogram2D(const ImageData& srcImage, const std::vector<float>& target1, float weight1, const std::vector<float>& target2, float weight2, const char* outputFileName)
{
	// 2D barycentric coordinates. They add up to 1.0.
	float u = 1.0f - (weight1 + weight2);
	float v = weight1;
	float w = weight2;

	// Do interpolation
	ImageData output = srcImage;
	for (size_t valueIndex = 0; valueIndex < output.width * output.height * 3; ++valueIndex)
		output.pixels[valueIndex] = srcImage.pixels[valueIndex] * u + target1[valueIndex] * v + target2[valueIndex] * w;

	// Save output image
	SaveFloatImage(output, outputFileName);
}

int main(int argc, char** argv)
{
	_mkdir("out");

	// Load the images
	ImageData srcImage;
	if (!LoadImageAsFloat(srcImage, "images/florida.png"))
	{
		printf("could not load images/florida.png");
		return 1;
	}

	ImageData imageDunes;
	if (!LoadImageAsFloat(imageDunes, "images/dunes.png"))
	{
		printf("could not load images/dunes.png");
		return 1;
	}

	ImageData imageTurtle;
	if (!LoadImageAsFloat(imageTurtle, "images/turtle.png"))
	{
		printf("could not load images/turtle.png");
		return 1;
	}

	ImageData imageBigCat;
	if (!LoadImageAsFloat(imageBigCat, "images/bigcat.png"))
	{
		printf("could not load images/bigcat.png");
		return 1;
	}

	// Calculate optimal transport from the source image to the other images
	std::vector<float> OTDunes;
	SlicedOptimalTransport(srcImage, imageDunes, OTDunes, "out/dunes.csv");

	std::vector<float> OTTurtle;
	SlicedOptimalTransport(srcImage, imageTurtle, OTTurtle, "out/turtle.csv");

	std::vector<float> OTBigCat;
	SlicedOptimalTransport(srcImage, imageBigCat, OTBigCat, "out/bigcat.csv");

	// Make results
	InterpolateColorHistogram1D(srcImage, OTDunes, 1.0f, "out/florida-dunes.png");
	InterpolateColorHistogram1D(srcImage, OTTurtle, 1.0f, "out/florida-turtle.png");
	InterpolateColorHistogram1D(srcImage, OTBigCat, 1.0f, "out/florida-bigcat.png");

	// Do 1d barycentric interpolation towards bigcat
	for (int i = 0; i < 3; ++i)
	{
		float alpha = float(i + 1) / 4;
		int percent = int(alpha * 100.0f);
		char fileName[1024];
		sprintf_s(fileName, "out/florida-bigcat_%i.png", percent);
		InterpolateColorHistogram1D(srcImage, OTBigCat, alpha, fileName);
	}

	// Do 2d barycentric interpolation towards turtle and dunes
	InterpolateColorHistogram2D(srcImage, OTTurtle, 0.0f, OTDunes, 0.33f, "out/florida-turtle_0_dunes_33.png");
	InterpolateColorHistogram2D(srcImage, OTTurtle, 0.0f, OTDunes, 0.66f, "out/florida-turtle_0_dunes_66.png");
	InterpolateColorHistogram2D(srcImage, OTTurtle, 0.33f, OTDunes, 0.0f, "out/florida-turtle_33_dunes_0.png");
	InterpolateColorHistogram2D(srcImage, OTTurtle, 0.66f, OTDunes, 0.0f, "out/florida-turtle_66_dunes_0.png");
	InterpolateColorHistogram2D(srcImage, OTTurtle, 0.33f, OTDunes, 0.66f, "out/florida-turtle_33_dunes_66.png");
	InterpolateColorHistogram2D(srcImage, OTTurtle, 0.66f, OTDunes, 0.33f, "out/florida-turtle_66_dunes_33.png");
	InterpolateColorHistogram2D(srcImage, OTTurtle, 0.33f, OTDunes, 0.33f, "out/florida-turtle_33_dunes_33.png");

	return 0;
}

/*
TODO:
! 300 lines of C++ code, and just stb for image reading and writing.
! mention in the post that you could do barycentric interpolation with multiple images. and could even go "outside of the simplex" with those coordinates to move away from histograms etc.
! mention that the sorting is the slowest part, per the profiler. same as the other post. could multithread it but :shrug:
! say how long it took, and what resolution image
! show graphs of csvs
* mention how you can do all the batches in parallel
* ! show example of 1d and 2d interpolation. both the images and the histograms
*/
