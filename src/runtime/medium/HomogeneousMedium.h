#pragma once

#include "Medium.h"

namespace IG {
class HomogeneousMedium : public Medium {
public:
    HomogeneousMedium(const std::string& name, const std::shared_ptr<SceneObject>& medium);
    void serialize(const SerializationInput& input) const override;

private:
    std::shared_ptr<SceneObject> mMedium;
};
} // namespace IG