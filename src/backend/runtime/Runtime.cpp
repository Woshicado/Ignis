#include "Runtime.h"
#include "Camera.h"
#include "Logger.h"
#include "driver/Configuration.h"
#include "jit.h"
#include "loader/Parser.h"

#include <chrono>

namespace IG {

static inline void setup_technique(LoaderOptions& lopts, const RuntimeOptions& opts)
{
	std::string tech_type;
	if (opts.OverrideTechnique.empty()) {
		const auto technique = lopts.Scene.technique();
		if (technique)
			tech_type = technique->pluginType();
		else
			tech_type = "path";
	} else {
		tech_type = opts.OverrideTechnique;
	}
	lopts.TechniqueType = tech_type;
}

static inline void setup_film(RuntimeRenderSettings& settings, const LoaderOptions& lopts, const RuntimeOptions& opts)
{
	IG_UNUSED(opts);

	const auto film = lopts.Scene.film();
	if (film) {
		const auto filmSize = film->property("size").getVector2(Vector2f(settings.FilmWidth, settings.FilmHeight));
		settings.FilmWidth	= filmSize.x();
		settings.FilmHeight = filmSize.y();
	}
}

static inline void setup_camera(RuntimeRenderSettings& settings, LoaderOptions& lopts, const RuntimeOptions& opts)
{
	// Extract camera type first
	std::string camera_type = "perspective";
	if (!opts.OverrideCamera.empty()) {
		camera_type = opts.OverrideCamera;
	} else {
		const auto camera = lopts.Scene.camera();
		if (camera)
			camera_type = camera->pluginType();
	}

	lopts.CameraType = camera_type;

	// Get initial location
	Transformf cameraTransform;
	const auto camera = lopts.Scene.camera();
	if (camera) {
		cameraTransform = camera->property("transform").getTransform();
		settings.FOV	= camera->property("fov").getNumber(settings.FOV);
	}

	settings.CameraEye = cameraTransform * Vector3f::Zero();
	settings.CameraDir = cameraTransform.linear().col(2);
	settings.CameraUp  = cameraTransform.linear().col(1);

	settings.TMin = camera->property("near_clip").getNumber(settings.TMin);
	settings.TMax = camera->property("far_clip").getNumber(settings.TMax);

	if (settings.TMax < settings.TMin)
		std::swap(settings.TMin, settings.TMax);
}

Runtime::Runtime(const std::filesystem::path& path, const RuntimeOptions& opts)
	: mInit(false)
	, mDevice(opts.Device)
	, mIteration(0)
	, mIsTrace(false)
	, mIsDebug(false)
	, mDebugMode(DebugMode::Normal)
{
	if (!mManager.init())
		throw std::runtime_error("Could not init modules!");

	LoaderOptions lopts;
	lopts.FilePath		= path;
	lopts.Configuration = targetToConfiguration(opts.DesiredTarget);

	// Parse scene file
	const auto startParser = std::chrono::high_resolution_clock::now();
	Parser::SceneParser parser;
	bool ok		= false;
	lopts.Scene = parser.loadFromFile(path, ok);
	if (!ok)
		throw std::runtime_error("Could not parse scene!");
	IG_LOG(L_DEBUG) << "Parsing scene took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startParser).count() / 1000.0f << " seconds" << std::endl;

	// Extract technique
	setup_technique(lopts, opts);

	// Extract film
	setup_film(mLoadedRenderSettings, lopts, opts);

	// Extract camera
	setup_camera(mLoadedRenderSettings, lopts, opts);

	// Check configuration
	const uint64 newConfig = mManager.checkConfiguration(lopts.Configuration);
	if (newConfig != lopts.Configuration) {
		IG_LOG(L_WARNING) << "Switched from "
						  << configurationToString(lopts.Configuration) << " to "
						  << configurationToString(newConfig) << std::endl;
	}
	lopts.Configuration = newConfig;

	LoaderResult result;
	if (!Loader::load(lopts, result))
		throw std::runtime_error("Could not load scene!");
	mDatabase = std::move(result.Database);

	IG_LOG(L_INFO) << "Loading configuration " << configurationToString(newConfig & IG_C_MASK_DEVICE) << std::endl;
	if (!mManager.load(newConfig & IG_C_MASK_DEVICE, mLoadedInterface))
		throw std::runtime_error("Error loading interface!");
	mConfiguration = newConfig;

	mIsDebug = lopts.TechniqueType == "debug";
	mIsTrace = lopts.CameraType == "list";

	IG_LOG(L_DEBUG) << "Ray Generation Shader:" << std::endl
					<< result.RayGenerationShader << std::endl;

	IG_LOG(L_DEBUG) << "Miss Shader:" << std::endl
					<< result.MissShader << std::endl;
	for (const auto& shader : result.HitShaders) {
		IG_LOG(L_DEBUG) << "Hit Shader:" << std::endl
						<< shader << std::endl;
	}

	RayGenerationShader = std::move(result.RayGenerationShader);
	MissShader			= std::move(result.MissShader);
	HitShaders			= std::move(result.HitShaders);

	// Force flush to zero mode for denormals
#if defined(__x86_64__) || defined(__amd64__) || defined(_M_X64)
	_mm_setcsr(_mm_getcsr() | (_MM_FLUSH_ZERO_ON | _MM_DENORMALS_ZERO_ON));
#endif
}

Runtime::~Runtime()
{
	if (mInit)
		shutdown();
}

void Runtime::step(const Camera& camera)
{
	IG_ASSERT(mInit, "Expected to be initialized!");

	if (mIsTrace) {
		IG_LOG(L_ERROR) << "Trying to use step() in a trace driver!" << std::endl;
		return;
	}

	DriverRenderSettings settings;
	for (int i = 0; i < 3; ++i)
		settings.eye[i] = camera.Eye(i);
	for (int i = 0; i < 3; ++i)
		settings.dir[i] = camera.Direction(i);
	for (int i = 0; i < 3; ++i)
		settings.up[i] = camera.Up(i);
	for (int i = 0; i < 3; ++i)
		settings.right[i] = camera.Right(i);
	settings.width		= camera.SensorWidth;
	settings.height		= camera.SensorHeight;
	settings.tmin		= camera.TMin;
	settings.tmax		= camera.TMax;
	settings.rays		= nullptr; // No artifical ray streams
	settings.device		= mDevice;
	settings.debug_mode = (uint32)mDebugMode;

	mLoadedInterface.RenderFunction(&settings, mIteration++);
}

void Runtime::trace(const std::vector<Ray>& rays, std::vector<float>& data)
{
	if (!mInit)
		return;

	if (!mIsTrace) {
		IG_LOG(L_ERROR) << "Trying to use trace() in a camera driver!" << std::endl;
		return;
	}

	DriverRenderSettings settings;
	settings.width	= rays.size();
	settings.height = 1;
	settings.rays	= rays.data();
	settings.device = mDevice;

	mLoadedInterface.RenderFunction(&settings, mIteration++);

	// Get result
	const float* data_ptr = getFramebuffer(0);
	data.resize(rays.size() * 3);
	std::memcpy(data.data(), data_ptr, sizeof(float) * rays.size() * 3);
}

const float* Runtime::getFramebuffer(int aov) const
{
	return mLoadedInterface.GetFramebufferFunction(aov);
}

void Runtime::clearFramebuffer(int aov)
{
	return mLoadedInterface.ClearFramebufferFunction(aov);
}

void Runtime::setup(uint32 framebuffer_width, uint32 framebuffer_height)
{
	DriverSetupSettings settings;
	settings.database			= &mDatabase;
	settings.framebuffer_width	= std::max(1u, framebuffer_width);
	settings.framebuffer_height = std::max(1u, framebuffer_height);

	IG_LOG(L_DEBUG) << "Init JIT compiling" << std::endl;
	ig_init_jit(mManager.getPath(mConfiguration & IG_C_MASK_DEVICE).generic_u8string());

	IG_LOG(L_DEBUG) << "Compiling ray generation shader" << std::endl;
	settings.ray_generation_shader = ig_compile_source(RayGenerationShader, "ig_ray_generation_shader");

	IG_LOG(L_DEBUG) << "Compiling miss shader" << std::endl;
	settings.miss_shader = ig_compile_source(MissShader, "ig_miss_shader");

	IG_LOG(L_DEBUG) << "Compiling hit shader" << std::endl;
	for (size_t i = 0; i < HitShaders.size(); ++i)
		settings.hit_shaders.push_back(ig_compile_source(HitShaders[i], "ig_hit_shader_" + std::to_string(i)));

	mLoadedInterface.SetupFunction(&settings);
	mInit = true;

	clearFramebuffer();
}

void Runtime::shutdown()
{
	mLoadedInterface.ShutdownFunction();
}

} // namespace IG