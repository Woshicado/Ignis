#pragma once

#include "CameraOrientation.h"
#include "LoaderOptions.h"
#include "RuntimeSettings.h"
#include "TechniqueInfo.h"
#include "device/Target.h"
#include "math/BoundingBox.h"
#include "table/SceneDatabase.h"

#include <any>
#include <filesystem>
#include <unordered_set>
#include <vector>

namespace IG {
// TODO: Refactor to LoaderEntity?
struct Entity {
    size_t ID;
    Transformf Transform;
    std::string Name;
    uint32 ShapeID;
    std::string BSDF;

    inline Eigen::Matrix<float, 3, 4> computeLocalMatrix() const
    {
        return Transform.inverse().matrix().block<3, 4>(0, 0);
    }
    inline Eigen::Matrix<float, 3, 4> computeGlobalMatrix() const
    {
        return Transform.matrix().block<3, 4>(0, 0);
    }
    inline Eigen::Matrix<float, 3, 3> computeGlobalNormalMatrix() const
    {
        return Transform.matrix().block<3, 3>(0, 0).inverse().transpose();
    }
};

/// A material is a combination of bsdf, entity (if the entity is emissive) and volume/medium interface
struct Material {
    std::string BSDF;
    int MediumInner = -1;
    int MediumOuter = -1;
    std::string Entity; // Empty if not emissive
    inline bool hasEmission() const { return !Entity.empty(); }
    inline bool hasMediumInterface() const { return MediumInner >= 0 || MediumOuter >= 0; }
};

inline bool operator==(const Material& a, const Material& b)
{
    return a.BSDF == b.BSDF && a.MediumInner == b.MediumInner && a.MediumOuter == b.MediumOuter && a.Entity == b.Entity;
}

constexpr size_t DefaultAlignment = sizeof(float) * 4;
class LoaderContext {
    IG_CLASS_NON_COPYABLE(LoaderContext);

public:
    LoaderContext();
    LoaderContext(LoaderContext&&);
    LoaderContext& operator=(LoaderContext&&);
    ~LoaderContext();

    LoaderOptions Options;

    std::unique_ptr<class LoaderLight> Lights;
    std::unique_ptr<class LoaderShape> Shapes;
    std::unique_ptr<class LoaderEntity> Entities;

    SceneDatabase Database;
    std::vector<TechniqueVariant> TechniqueVariants; // TODO: Refactor this out, as no loader requires this, but will produce it...
    IG::TechniqueInfo TechniqueInfo;
    IG::CameraOrientation CameraOrientation;

    size_t CurrentTechniqueVariant;
    [[nodiscard]] inline const IG::TechniqueVariantInfo CurrentTechniqueVariantInfo() const { return TechniqueInfo.Variants.at(CurrentTechniqueVariant); }

    ParameterSet LocalRegistry; // Current local registry for given shader
    inline void resetRegistry()
    {
        LocalRegistry = ParameterSet();
    }

    std::unordered_map<std::string, std::any> ExportedData; // Cache with already exported data and auxillary info

    // TODO: Refactor this into LoaderEntity?
    std::unordered_map<std::string, Entity> EmissiveEntities;
    std::vector<Material> Materials;

    BoundingBox SceneBBox;
    float SceneDiameter = 0.0f;

    size_t EntityCount;

    std::filesystem::path handlePath(const std::filesystem::path& path, const Parser::Object& obj) const;

    std::unordered_map<std::string, size_t> RegisteredResources;
    inline size_t registerExternalResource(const std::filesystem::path& path)
    {
        // TODO: Ensure canonical paths?
        auto it = RegisteredResources.find(path.generic_u8string());
        if (it != RegisteredResources.end())
            return it->second;
        const size_t id = RegisteredResources.size();

        return RegisteredResources[path.generic_u8string()] = id;
    }

    [[nodiscard]] inline std::vector<std::string> generateResourceMap() const
    {
        std::vector<std::string> map(RegisteredResources.size());
        for (const auto& p : RegisteredResources) {
            IG_ASSERT(p.second < map.size(), "Access should be always in bound");
            map[p.second] = p.first;
        }
        return map;
    }

    bool HasError = false;

    /// Use this function to mark the loading process as failed
    inline void signalError()
    {
        HasError = true;
    }
};

} // namespace IG