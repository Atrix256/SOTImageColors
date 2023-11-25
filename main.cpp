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

using Vec3 = float[3];

struct ImageData
{
	int width = 0;
	int height = 0;
	std::vector<float> pixels;
};

struct TargetImage
{
	const char* fileName = nullptr;
	float weight = 0.0f;
	ImageData image;
	std::vector<float> sourceDelta;
};

struct Settings
{
	const char* srcImageFileName = nullptr;
	ImageData srcimage;
	std::vector<TargetImage> targetImages;
};

std::mt19937 GetRNG()
{
	#if DETERMINISTIC()
	std::mt19937 ret;
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

void SlicedOptimalTransport(const ImageData& source, TargetImage& targetImage)
{
	printf("\n%s (weight %0.2f)\n\n", targetImage.fileName, targetImage.weight);

	const uint32_t c_numPixels = source.width * source.height;

	std::mt19937 rng = GetRNG();
	std::normal_distribution<float> normalDist(0.0f, 1.0f);

	ImageData current = source;

	std::vector<uint32_t> currentSorted(c_numPixels);
	std::vector<uint32_t> targetSorted(c_numPixels);

	std::vector<float> currentProjections(c_numPixels);
	std::vector<float> targetProjections(c_numPixels);

	std::vector<float> batchDirections(c_numPixels * 3);

	for (int iteration = 0; iteration < c_numIterations; ++iteration)
	{
		std::fill(batchDirections.begin(), batchDirections.end(), 0.0f);

		for (int batchIndex = 0; batchIndex < c_batchSize; ++batchIndex)
		{
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
				currentSorted[i] = (uint32_t)i;
				targetSorted[i] = (uint32_t)i;

				currentProjections[i] =
					direction[0] * current.pixels[i * 3 + 0] +
					direction[1] * current.pixels[i * 3 + 1] +
					direction[2] * current.pixels[i * 3 + 2];

				targetProjections[i] =
					direction[0] * targetImage.image.pixels[i * 3 + 0] +
					direction[1] * targetImage.image.pixels[i * 3 + 1] +
					direction[2] * targetImage.image.pixels[i * 3 + 2];
			}

			// sort
			std::sort(currentSorted.begin(), currentSorted.end(),
				[&] (uint32_t a, uint32_t b)
				{
					return currentProjections[a] < currentProjections[b];
				}
			);

			std::sort(targetSorted.begin(), targetSorted.end(),
				[&](uint32_t a, uint32_t b)
				{
					return targetProjections[a] < targetProjections[b];
				}
			);

			// update batchDirections
			for (size_t i = 0; i < c_numPixels; ++i)
			{
				float projDiff = targetProjections[targetSorted[i]] - currentProjections[currentSorted[i]];

				batchDirections[currentSorted[i] * 3 + 0] += direction[0] * projDiff;
				batchDirections[currentSorted[i] * 3 + 1] += direction[1] * projDiff;
				batchDirections[currentSorted[i] * 3 + 2] += direction[2] * projDiff;
			}
		}

		// update current
		float totalDistance = 0.0f;
		for (size_t i = 0; i < c_numPixels; ++i)
		{
			float adjust[3] = {
				batchDirections[currentSorted[i] * 3 + 0] / float(c_batchSize),
				batchDirections[currentSorted[i] * 3 + 1] / float(c_batchSize),
				batchDirections[currentSorted[i] * 3 + 2] / float(c_batchSize)
			};

			current.pixels[currentSorted[i] * 3 + 0] += adjust[0];
			current.pixels[currentSorted[i] * 3 + 1] += adjust[1];
			current.pixels[currentSorted[i] * 3 + 2] += adjust[2];

			totalDistance += std::sqrt(adjust[0] * adjust[0] + adjust[1] * adjust[1] + adjust[2] * adjust[2]);
		}

		printf("[%i] %f\n", iteration, totalDistance);
	}

	// Calculate deltas
	targetImage.sourceDelta.resize(c_numPixels * 3);
	for (size_t i = 0; i < c_numPixels * 3; ++i)
		targetImage.sourceDelta[i] = current.pixels[i] - source.pixels[i];
}

void InterpolateColorHistogram(Settings& settings, const char* outputFileName)
{
	printf("==================================\n%s\n==================================\n", outputFileName);

	// load up the source image
	LoadImageAsFloat(settings.srcImageFileName, settings.srcimage);

	// do SOT on each target image
	for (size_t index = 0; index < settings.targetImages.size(); ++index)
	{
		// load the target image and verify it's compatible
		TargetImage& targetImage = settings.targetImages[index];
		LoadImageAsFloat(targetImage.fileName, targetImage.image);
		if (targetImage.image.width != settings.srcimage.width || targetImage.image.height != settings.srcimage.height)
		{
			printf("ERROR: image %s is %ix%i, but should be %ix%i like %s.\n",
				targetImage.fileName, targetImage.image.width, targetImage.image.height,
				settings.srcimage.width, settings.srcimage.height, settings.srcImageFileName);
			return;
		}

		// do optimal transport to get the per pixel delta to get from the source image to the target image
		SlicedOptimalTransport(settings.srcimage, settings.targetImages[index]);
	}

	// Do barycentric interpolation
	ImageData output = settings.srcimage;
	float baseWeight = 1.0f;
	for (const TargetImage& targetImage : settings.targetImages)
	{
		for (size_t valueIndex = 0; valueIndex < output.width * output.height * 3; ++valueIndex)
			output.pixels[valueIndex] += targetImage.sourceDelta[valueIndex] * targetImage.weight;
	}

	// Save output image
	SaveFloatImage(output, outputFileName);
}

int main(int argc, char** argv)
{
	_mkdir("out");

	system("python MakeHistogram.py images/florida.png out/florida.histogram.png 12000");
	system("python MakeHistogram.py images/bigcat.png out/bigcat.histogram.png 12000");
	system("python MakeHistogram.py images/dunes.png out/dunes.histogram.png 12000");
	system("python MakeHistogram.py images/turtle.png out/turtle.histogram.png 12000");

	{
		Settings settings;
		settings.srcImageFileName = "images/florida.png";
		settings.targetImages.push_back({"images/dunes.png", 1.0f});
		InterpolateColorHistogram(settings, "out/florida-dunes.png");

		system("python MakeHistogram.py out/florida-dunes.png out/florida-dunes.histogram.png 12000");
	}

	{
		Settings settings;
		settings.srcImageFileName = "images/florida.png";
		settings.targetImages.push_back({ "images/turtle.png", 1.0f });
		InterpolateColorHistogram(settings, "out/florida-turtle.png");

		system("python MakeHistogram.py out/florida-turtle.png out/florida-turtle.histogram.png 12000");
	}

	{
		Settings settings;
		settings.srcImageFileName = "images/florida.png";
		settings.targetImages.push_back({ "images/bigcat.png", 1.0f });
		InterpolateColorHistogram(settings, "out/florida-bigcat.png");

		system("python MakeHistogram.py out/florida-bigcat.png out/florida-bigcat.histogram.png 12000");
	}

	return 0;
}

/*
TODO:
* make a csv of total movement over time and graph it to see if it's enough steps
* clean up code?
* optimize code? omp?
* make an interpolation example with 2 images
* make an interpolation example with 3 images
*/
