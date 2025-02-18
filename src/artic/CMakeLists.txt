set(ARTIC_MAIN_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/bsdf/common.art
    ${CMAKE_CURRENT_SOURCE_DIR}/bsdf/conductor.art
    ${CMAKE_CURRENT_SOURCE_DIR}/bsdf/dielectric.art
    ${CMAKE_CURRENT_SOURCE_DIR}/bsdf/diffuse.art
    ${CMAKE_CURRENT_SOURCE_DIR}/bsdf/djmeasured.art
    ${CMAKE_CURRENT_SOURCE_DIR}/bsdf/klems.art
    ${CMAKE_CURRENT_SOURCE_DIR}/bsdf/map.art
    ${CMAKE_CURRENT_SOURCE_DIR}/bsdf/mix.art
    ${CMAKE_CURRENT_SOURCE_DIR}/bsdf/phong.art
    ${CMAKE_CURRENT_SOURCE_DIR}/bsdf/plastic.art
    ${CMAKE_CURRENT_SOURCE_DIR}/bsdf/principled.art
    ${CMAKE_CURRENT_SOURCE_DIR}/bsdf/tensortree.art
    ${CMAKE_CURRENT_SOURCE_DIR}/camera/common.art
    ${CMAKE_CURRENT_SOURCE_DIR}/camera/fishlens.art
    ${CMAKE_CURRENT_SOURCE_DIR}/camera/orthogonal.art
    ${CMAKE_CURRENT_SOURCE_DIR}/camera/perspective.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/bbox.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/cdf.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/color.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/common.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/cpu_common.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/fresnel.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/gpu_common.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/interval.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/math.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/matrix.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/microfacet.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/print.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/random.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/sampling.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/shading.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/sort.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/stats.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/triangle.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/vector.art
    ${CMAKE_CURRENT_SOURCE_DIR}/core/warp.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/accumulator.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/aov.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/bsdf.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/buffer_cpu.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/buffer_gpu.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/buffer.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/camera.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/data.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/device.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/driver.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/emitter.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/entity.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/filter.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/image.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/light.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/mapping_cpu.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/mapping_gpu.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/material.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/medium.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/parallel_reduce.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/payload.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/pdf.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/phase.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/pipeline.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/pointmapper.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/registry.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/scene.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/settings.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/shading_context.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/shape.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/streams.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/surface_element.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/technique.art
    ${CMAKE_CURRENT_SOURCE_DIR}/driver/tonemap.art
    ${CMAKE_CURRENT_SOURCE_DIR}/entrypoints/bake.art
    ${CMAKE_CURRENT_SOURCE_DIR}/entrypoints/imageinfo.art
    ${CMAKE_CURRENT_SOURCE_DIR}/entrypoints/render.art
    ${CMAKE_CURRENT_SOURCE_DIR}/entrypoints/tonemap.art
    ${CMAKE_CURRENT_SOURCE_DIR}/light/area.art
    ${CMAKE_CURRENT_SOURCE_DIR}/light/cie.art
    ${CMAKE_CURRENT_SOURCE_DIR}/light/directional.art
    ${CMAKE_CURRENT_SOURCE_DIR}/light/env.art
    ${CMAKE_CURRENT_SOURCE_DIR}/light/light_hierarchy.art
    ${CMAKE_CURRENT_SOURCE_DIR}/light/light_selector.art
    ${CMAKE_CURRENT_SOURCE_DIR}/light/point.art
    ${CMAKE_CURRENT_SOURCE_DIR}/light/spot.art
    ${CMAKE_CURRENT_SOURCE_DIR}/light/sun.art
    ${CMAKE_CURRENT_SOURCE_DIR}/medium/common.art
    ${CMAKE_CURRENT_SOURCE_DIR}/medium/homogeneous.art
    ${CMAKE_CURRENT_SOURCE_DIR}/phase/common.art
    ${CMAKE_CURRENT_SOURCE_DIR}/phase/henyeygreenstein.art
    ${CMAKE_CURRENT_SOURCE_DIR}/phase/uniform.art
    ${CMAKE_CURRENT_SOURCE_DIR}/sampler/pixel_sampler.art
    ${CMAKE_CURRENT_SOURCE_DIR}/shapes/sphere.art
    ${CMAKE_CURRENT_SOURCE_DIR}/shapes/trimesh.art
    ${CMAKE_CURRENT_SOURCE_DIR}/technique/aotracer.art
    ${CMAKE_CURRENT_SOURCE_DIR}/technique/camera_check.art
    ${CMAKE_CURRENT_SOURCE_DIR}/technique/debugtracer.art
    ${CMAKE_CURRENT_SOURCE_DIR}/technique/infobuffer.art
    ${CMAKE_CURRENT_SOURCE_DIR}/technique/lighttracer.art
    ${CMAKE_CURRENT_SOURCE_DIR}/technique/lightvisibility.art
    ${CMAKE_CURRENT_SOURCE_DIR}/technique/pathtracer.art
    ${CMAKE_CURRENT_SOURCE_DIR}/technique/photonmapper.art
    ${CMAKE_CURRENT_SOURCE_DIR}/technique/volpathtracer.art
    ${CMAKE_CURRENT_SOURCE_DIR}/technique/wireframe.art
    ${CMAKE_CURRENT_SOURCE_DIR}/texture/brick.art
    ${CMAKE_CURRENT_SOURCE_DIR}/texture/bump.art
    ${CMAKE_CURRENT_SOURCE_DIR}/texture/checkerboard.art
    ${CMAKE_CURRENT_SOURCE_DIR}/texture/common.art
    ${CMAKE_CURRENT_SOURCE_DIR}/texture/image.art
    ${CMAKE_CURRENT_SOURCE_DIR}/texture/noise.art
    ${CMAKE_CURRENT_SOURCE_DIR}/traversal/bvh.art
    ${CMAKE_CURRENT_SOURCE_DIR}/traversal/intersection.art
    ${CMAKE_CURRENT_SOURCE_DIR}/traversal/mapping_cpu.art
    ${CMAKE_CURRENT_SOURCE_DIR}/traversal/mapping_gpu.art
    ${CMAKE_CURRENT_SOURCE_DIR}/traversal/ray.art
    ${CMAKE_CURRENT_SOURCE_DIR}/traversal/stack.art
)

set(ARTIC_MAIN_SRC ${ARTIC_MAIN_SRC} PARENT_SCOPE)

anydsl_runtime_wrap(DISCARD_TMP_OBJS
    NAME "interface"
    FRONTEND "artic"
    FILES ${ARTIC_MAIN_SRC} dummy_main.art dummy_extra.art
    INTERFACE ${CMAKE_BINARY_DIR}/generated_interface)
add_custom_target(artic_c_interface ALL DEPENDS ${CMAKE_BINARY_DIR}/generated_interface.h)
