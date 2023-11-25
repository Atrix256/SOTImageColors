#define _CRT_SECURE_NO_WARNINGS // for stb

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <random>
#include <vector>

struct ImageData
{
	int width = 0;
	int height = 0;
	int channels = 0;
	std::vector<float> pixels;
};

struct TargetImage
{
	const char* fileName = nullptr;
	float weight = 0.0f;
	ImageData image;
};

struct Settings
{
	const char* srcImageFileName = nullptr;
	ImageData srcimage;
	std::vector<TargetImage> targetImages;
};

#define DETERMINISTIC() true

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

bool LoadImageAsFloat(const char* fileName, int numChannels, ImageData& imageData)
{
	int c;
	stbi_uc* pixelsU8 = stbi_load(fileName, &imageData.width, &imageData.height, &c, numChannels);
	if (!pixelsU8)
		return false;

	imageData.channels = numChannels;
	imageData.pixels.resize(imageData.width * imageData.height * imageData.channels);

	for (size_t i = 0; i < imageData.width * imageData.height * imageData.channels; ++i)
		imageData.pixels[i] = float(pixelsU8[i]) / 255.0f;

	stbi_image_free(pixelsU8);
	return true;
}

void InterpolateColorHistogram(Settings& settings, const char* outputFileName)
{
	// load up the source image
	LoadImageAsFloat(settings.srcImageFileName, 3, settings.srcimage);

	// do SOT on each target image
	for (size_t index = 0; index < settings.targetImages.size(); ++index)
	{

	}

	//std::mt19937 rng = GetRNG();
	//std::normal_distribution<float> normalDist(0.0f, 1.0f);


}

int main(int argc, char** argv)
{
	// make florida1.png use the colors from turtle.png
	{
		Settings settings;
		settings.srcImageFileName = "images/florida1.png";
		settings.targetImages.push_back({"images/turtle.png", 1.0f});
		InterpolateColorHistogram(settings, "out/test1.png");
	}

	return 0;
}