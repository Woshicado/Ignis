#pragma once

#include "GeneratorContext.h"

namespace IG {
struct GeneratorBSDF {
	static std::string extract(const std::shared_ptr<Loader::Object>& bsdf, const GeneratorContext& ctx);
};
} // namespace IG