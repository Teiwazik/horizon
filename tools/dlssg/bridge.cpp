// Native Vulkan bridge for the isolated Horizon experiment.
#include "bridge.h"
#include <vulkan/vulkan.h>
#include "nvsdk_ngx_vk.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <mutex>
static decltype(&NVSDK_NGX_VULKAN_EvaluateFeature) direct_eval;
static NVSDK_NGX_Result evaluate_c(VkCommandBuffer cmd, const NVSDK_NGX_Handle *handle,
    const NVSDK_NGX_Parameter *params, PFN_NVSDK_NGX_ProgressCallback_C) {
    return direct_eval(cmd,handle,params,nullptr);
}
#define NVSDK_NGX_VULKAN_EvaluateFeature_C evaluate_c
#include "nvsdk_ngx_helpers_dlssg_vk.h"
#undef NVSDK_NGX_VULKAN_EvaluateFeature_C
#include "nvsdk_ngx_helpers_vk.h"
static std::mutex mutex;
static VkDevice sdk_device{};
static void *sdk_module{};
static bool runtime_initialized{};
struct HookScope {
    using Scope=void (*)(int);
    Scope scope=reinterpret_cast<Scope>(dlsym(RTLD_DEFAULT,"horizon_dlssg_scope"));
    HookScope() { if (scope) scope(1); }
    ~HookScope() { if (scope) scope(0); }
};
static void logger(const char *message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    fprintf(stderr,"Horizon DLSS-G: %s\n",message);
}
struct Context {
    void *module{}; VkDevice device{}; NVSDK_NGX_Parameter *params{}; NVSDK_NGX_Handle *handle{};
    unsigned evaluations{};
    decltype(&NVSDK_NGX_VULKAN_ReleaseFeature) release{};
    decltype(&NVSDK_NGX_VULKAN_DestroyParameters) destroy_params{};
};
// The C ABI variant of the SDK's project identifier initialization; its declaration lives
// only in the vendor header's C path (the same declaration the prototype's initialize.cpp uses).
extern "C" {
NVSDK_NGX_Result NVSDK_NGX_VULKAN_Init_with_ProjectID(
    const char *ProjectId, NVSDK_NGX_EngineType EngineType, const char *EngineVersion,
    const wchar_t *ApplicationDataPath, VkInstance VkInstanceIn, VkPhysicalDevice VkPhysicalDeviceIn,
    VkDevice VkDeviceIn, PFN_vkGetInstanceProcAddr get_instance_addressing,
    PFN_vkGetDeviceProcAddr get_device_addressing, const NVSDK_NGX_FeatureCommonInfo *FeatureInfo,
    NVSDK_NGX_Version FunctionVersion);
}
static NVSDK_NGX_Resource_VK resource(const HorizonFgImage &image,bool rw) {
    return NVSDK_NGX_Create_ImageView_Resource_VK(reinterpret_cast<VkImageView>(image.view),
        reinterpret_cast<VkImage>(image.image),{image.aspect,0,1,0,1},VkFormat(image.format),image.width,image.height,rw);
}
#ifndef PROJECT_ID_TEXT
#define PROJECT_ID_TEXT "6f1c2b8e-4d3a-4e9b-a7c5-31d08e2f9b64"
#endif
extern "C" void *horizon_fg_create(uint64_t instance,uint64_t physical,uint64_t device,
        uint64_t command,uint32_t width,uint32_t height) {
    std::lock_guard lock(mutex);
    HookScope hooks;
    // NGX retains SDK/allocator/logging callbacks after ReleaseFeature. Keep this
    // bridge mapped while those process-global callbacks can still be invoked.
    static void *pinned_bridge;
    if (!pinned_bridge) {
        Dl_info own{};
        if (!dladdr(reinterpret_cast<void *>(&horizon_fg_create), &own) ||
            !(pinned_bridge=dlopen(own.dli_fname,RTLD_NOW|RTLD_NODELETE))) {
            fprintf(stderr,"Horizon DLSS-G: could not pin bridge callbacks\n");
            return nullptr;
        }
    }
    auto context=new Context;
    context->device=reinterpret_cast<VkDevice>(device);
    const char *path=getenv("HORIZON_DLSSG_RUNTIME");
    std::string default_runtime=std::string(getenv("DLSS_SDK")?getenv("DLSS_SDK"):"")+"/lib/Linux_x86_64/rel/libnvidia-ngx-dlssg.so.310.9.16";
    const char *runtime_full=path?path:default_runtime.c_str();
    context->module=dlopen(runtime_full,RTLD_NOW|RTLD_LOCAL);
    if (!context->module) {fprintf(stderr,"Horizon DLSS-G: %s\n",dlerror());delete context;return nullptr;}
    auto init=reinterpret_cast<decltype(&NVSDK_NGX_VULKAN_Init_Ext2)>(dlsym(context->module,"NVSDK_NGX_VULKAN_Init_Ext2"));
    auto create=reinterpret_cast<decltype(&NVSDK_NGX_VULKAN_CreateFeature)>(dlsym(context->module,"NVSDK_NGX_VULKAN_CreateFeature"));
    direct_eval=reinterpret_cast<decltype(direct_eval)>(dlsym(context->module,"NVSDK_NGX_VULKAN_EvaluateFeature"));
    context->release=reinterpret_cast<decltype(context->release)>(dlsym(context->module,"NVSDK_NGX_VULKAN_ReleaseFeature"));
    if (!init||!create||!direct_eval||!context->release) {delete context;return nullptr;}
    // The SDK parameter table is per statically linked SDK copy, so follow the exact
    // sequence the working prototype proved: initialize through this copy's own
    // Init_with_ProjectID (the running game's SR path uses a different SDK copy, so its
    // table is unavailable), then allocate the parameter object from it, attach the
    // runtime through Init_Ext2, and create the feature with the runtime's own exports.
    NVSDK_NGX_FeatureCommonInfo common{};
    // Wide string directory of the runtime .so for the SDK's search path: strip the file
    // name and widen to Linux wchar_t (32-bit) character by character.
    std::string ansi{runtime_full};
    auto slash=ansi.find_last_of('/');
    if (slash!=std::string::npos) ansi.resize(slash);
    std::wstring dir(ansi.begin(),ansi.end());
    const wchar_t *dir_path=dir.c_str();
    const wchar_t *paths[] = {dir_path};
    common.PathListInfo.Path=paths; common.PathListInfo.Length=1;
    common.LoggingInfo.LoggingCallback=logger;
    common.LoggingInfo.MinimumLoggingLevel=NVSDK_NGX_LOGGING_LEVEL_ON;
    common.LoggingInfo.DisableOtherLoggingSinks=1;
    auto status=NVSDK_NGX_Result_Success;
    if (!sdk_device) {
    status=NVSDK_NGX_VULKAN_Init_with_ProjectID(PROJECT_ID_TEXT,NVSDK_NGX_ENGINE_TYPE_CUSTOM,
        "bridge",L"/tmp",reinterpret_cast<VkInstance>(instance),reinterpret_cast<VkPhysicalDevice>(physical),
        reinterpret_cast<VkDevice>(device),vkGetInstanceProcAddr,vkGetDeviceProcAddr,&common,
        NVSDK_NGX_Version_API);
    if (NVSDK_NGX_FAILED(status)) {fprintf(stderr,"Horizon DLSS-G: SDK init failed: %x\n",status);delete context;return nullptr;}
    sdk_device=context->device;
    sdk_module=context->module;
    }
    context->destroy_params=&NVSDK_NGX_VULKAN_DestroyParameters;
    status=NVSDK_NGX_VULKAN_AllocateParameters(&context->params);
    if (NVSDK_NGX_FAILED(status)||!context->params) {fprintf(stderr,"Horizon DLSS-G: parameters failed: %x\n",status);delete context;return nullptr;}
    context->params->Set("Log.Callback",reinterpret_cast<void*>(logger));
    context->params->Set("Minimum.Logging.Level",1u);
    context->params->Set("Disable.Other.Logging.Sinks",1u);
    if (!runtime_initialized) status=init(0,L"/tmp",reinterpret_cast<VkInstance>(instance),reinterpret_cast<VkPhysicalDevice>(physical),
                context->device,vkGetInstanceProcAddr,vkGetDeviceProcAddr,NVSDK_NGX_Version_API,context->params);
    if (NVSDK_NGX_SUCCEED(status)) {
        runtime_initialized=true;
        context->params->Set(NVSDK_NGX_DLSSG_Parameter_Width,width);
        context->params->Set(NVSDK_NGX_DLSSG_Parameter_Height,height);
        context->params->Set(NVSDK_NGX_DLSSG_Parameter_BackbufferFormat,unsigned(VK_FORMAT_R8G8B8A8_UNORM));
        status=create(reinterpret_cast<VkCommandBuffer>(command),NVSDK_NGX_Feature_FrameGeneration,context->params,&context->handle);
    }
    auto failures=reinterpret_cast<unsigned (*)()>(dlsym(RTLD_DEFAULT,"dlssg_prototype_kernel_failures"));
    if (NVSDK_NGX_FAILED(status)||!context->handle||(failures&&failures())) {
        fprintf(stderr,"Horizon DLSS-G: feature creation failed: %x\n",status);
        if(context->handle)context->release(context->handle);
        if(context->params)NVSDK_NGX_VULKAN_DestroyParameters(context->params);delete context;return nullptr;
    }
    fprintf(stderr,"Horizon DLSS-G: native frame generation ready at %ux%u\n",width,height);
    return context;
}
extern "C" uint32_t horizon_fg_evaluate(void *opaque,uint64_t command,const HorizonFgFrame *frame) {
    std::lock_guard lock(mutex);HookScope hooks;auto context=static_cast<Context*>(opaque);
    auto color=resource(frame->color,false),depth=resource(frame->depth,false),motion=resource(frame->motion,false),output=resource(frame->output,true);
    NVSDK_NGX_VK_DLSSG_Eval_Params resources{};
    resources.pBackbuffer=&color;resources.pHudless=&color;resources.pDepth=&depth;
    resources.pMVecs=&motion;resources.pOutputInterpFrame=&output;
    NVSDK_NGX_DLSSG_Opt_Eval_Params options{};
    memcpy(options.cameraViewToClip,frame->projection,64);
    memcpy(options.clipToCameraView,frame->projection_inverse,64);
    memcpy(options.clipToPrevClip,frame->reprojection,64);
    memcpy(options.prevClipToClip,frame->reprojection_inverse,64);
    for(int i=0;i<4;i++) options.clipToLensClip[i][i]=1;
    options.jitterOffset[0]=frame->jitter[0];options.jitterOffset[1]=frame->jitter[1];
    options.mvecScale[0]=1.f/frame->motion.width;options.mvecScale[1]=1.f/frame->motion.height;
    options.multiFrameCount=frame->multi_frame_count?frame->multi_frame_count:1;
    options.multiFrameIndex=frame->multi_frame_index?frame->multi_frame_index:1;
    options.cameraUp[1]=1;options.cameraRight[0]=1;options.cameraFwd[2]=-1;
    options.cameraNear=frame->near_plane;options.cameraFar=frame->far_plane;
    options.cameraFOV=frame->fov_y;options.cameraAspectRatio=float(frame->color.width)/frame->color.height;
    options.depthInverted=frame->depth_inverted!=0;options.cameraMotionIncluded=true;options.reset=frame->reset;
    options.backbufferSubrectSize={frame->color.width,frame->color.height};
    options.outputInterpSubrectSize={frame->output.width,frame->output.height};
    options.depthSubrectSize={frame->depth.width,frame->depth.height};
    options.mvecsSubrectSize={frame->motion.width,frame->motion.height};
    options.hudLessSubrectSize=options.backbufferSubrectSize;
    auto status=NGX_VK_EVALUATE_DLSSG(reinterpret_cast<VkCommandBuffer>(command),context->handle,context->params,&resources,&options);
    auto failures=reinterpret_cast<unsigned (*)()>(dlsym(RTLD_DEFAULT,"dlssg_prototype_kernel_failures"));
    if (failures && failures()) status=NVSDK_NGX_Result_Fail;
    if (context->evaluations++ < 9) fprintf(stderr,"Horizon DLSS-G: evaluate #%u result=%x reset=%u count=%u index=%u\n",context->evaluations,status,frame->reset,options.multiFrameCount,options.multiFrameIndex);
    if (NVSDK_NGX_FAILED(status)) fprintf(stderr,"Horizon DLSS-G: evaluation failed: %x\n",status);
    return unsigned(status);
}
extern "C" void horizon_fg_destroy(void *opaque) {
    if (!opaque)return;
    std::lock_guard lock(mutex);HookScope hooks;auto context=static_cast<Context*>(opaque);
    vkDeviceWaitIdle(context->device);
    if (context->handle)context->release(context->handle);
    if (context->params)context->destroy_params(context->params);
    // The process's NGX core owns shutdown; keep the feature module loaded until then.
    delete context;
}

extern "C" uint32_t horizon_fg_abi_version() { return 2; }

extern "C" void horizon_fg_shutdown() {
    std::lock_guard lock(mutex);
    HookScope hooks;
    if (!sdk_device) return;
    vkDeviceWaitIdle(sdk_device);
    auto shutdown=reinterpret_cast<decltype(&NVSDK_NGX_VULKAN_Shutdown)>(dlsym(sdk_module,"NVSDK_NGX_VULKAN_Shutdown"));
    if (runtime_initialized && shutdown) shutdown();
    // The process NGX core is shared with SR. The renderer shuts that core down
    // once, through its SR SDK, after this direct FG module has stopped.
    sdk_device=VK_NULL_HANDLE;
    sdk_module=nullptr;
    runtime_initialized=false;
    fprintf(stderr,"Horizon DLSS-G: SDK shutdown complete\n");
}
