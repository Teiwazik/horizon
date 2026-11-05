#pragma once
#include <stdint.h>
struct HorizonFgImage {
    uint64_t image, view;
    uint32_t format, width, height, aspect;
};
struct HorizonFgFrame {
    struct HorizonFgImage color, depth, motion, output;
    float projection[16], projection_inverse[16], reprojection[16], reprojection_inverse[16], jitter[2];
    float near_plane, far_plane, fov_y;
    uint32_t reset;
    // 1 when the depth buffer holds 1.0 at the near plane (reversed-Z, Horizon's convention).
    uint32_t depth_inverted;
    uint32_t multi_frame_count, multi_frame_index;
};
#ifdef __cplusplus
extern "C" {
#endif
uint32_t horizon_fg_abi_version(void);
void *horizon_fg_create(uint64_t instance, uint64_t physical, uint64_t device,
                       uint64_t command, uint32_t width, uint32_t height);
uint32_t horizon_fg_evaluate(void *context, uint64_t command, const struct HorizonFgFrame *frame);
void horizon_fg_destroy(void *context);
void horizon_fg_shutdown(void);
#ifdef __cplusplus
}
#endif
