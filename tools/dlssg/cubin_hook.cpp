// Isolated Vulkan CUDA-module hook. Original module bytes are dumped for diagnostics.
// Optional replacements are confined to this process; no driver/library files are patched.
#define VK_ENABLE_BETA_EXTENSIONS
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iterator>
#include <string>
#include <algorithm>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <atomic>

static PFN_vkCreateCuModuleNVX real_create;
static std::atomic<unsigned> sequence;
static std::atomic<unsigned> failures;
// NVIDIA can keep compiler jobs alive into global library finalization. These
// uploaded blobs must outlive those workers; let the OS reclaim them at exit.
static auto *kept = new std::unordered_set<std::string>;
static auto *kept_mutex = new std::mutex;
extern "C" unsigned dlssg_prototype_kernel_failures() { return failures.load(); }
static VkResult create_module(VkDevice device, const VkCuModuleCreateInfoNVX *info,
                             const VkAllocationCallbacks *alloc, VkCuModuleNVX *out) {
    Dl_info caller{};
    dladdr(__builtin_return_address(0), &caller);
    if (getenv("HORIZON_DLSSG_TRACE_CALLERS")) fprintf(stderr,"cubin caller=%s\n",caller.dli_fname?caller.dli_fname:"unknown");
    static auto in_scope=reinterpret_cast<int (*)()>(dlsym(RTLD_DEFAULT,"horizon_dlssg_in_scope"));
    bool fg=(in_scope && in_scope()) || (caller.dli_fname && strstr(caller.dli_fname,"libnvidia-ngx-dlssg"));
    if (!fg) return real_create(device,info,alloc,out);
    auto data = static_cast<const unsigned char *>(info->pData);
    unsigned serial = sequence++;
    char file[512]="disabled";
    if (const char *folder=getenv("HORIZON_DLSSG_DUMP")) {
        snprintf(file,sizeof(file),"%s/module-%03u.bin",folder,serial);
        std::ofstream dump(file,std::ios::binary);
        dump.write(reinterpret_cast<const char *>(data),info->dataSize);
    }
    std::vector<unsigned char> changed(data, data + info->dataSize);
    const char *mode = getenv("DLSSG_PROTOTYPE_KERNELS");
    if (mode && !strcmp(mode, "retarget")) {
        // This is the same PTX target retargeting described by dlssg-open's README.
        // The two ASCII digits have equal encoded size, including in literal LZ4 runs.
        const std::string needle = ".target sm_89";
        size_t replacements = 0;
        for (auto at = changed.begin(); (at = std::search(at, changed.end(), needle.begin(), needle.end())) != changed.end(); ++at) {
            at[needle.size() - 1] = '6'; replacements++;
        }
        // A fatbin's PTX record must agree with the rewritten .target literal.
        if (changed.size() >= 16 && !memcmp(changed.data(), "\x50\xed\x55\xba", 4)) {
            size_t pos = 16;
            while (pos + 64 <= changed.size()) {
                uint16_t kind; uint32_t header, arch; uint64_t padded;
                memcpy(&kind, changed.data()+pos, 2); memcpy(&header, changed.data()+pos+4, 4);
                memcpy(&padded, changed.data()+pos+8, 8); memcpy(&arch, changed.data()+pos+28, 4);
                if (header < 64 || header > changed.size()-pos || padded > changed.size()-pos-header) break;
                if (kind == 1 && arch == 89 && replacements) {
                    arch = 86; memcpy(changed.data()+pos+28, &arch, 4);
                }
                pos += header + padded;
            }
        }
        // Match exact original bytes rather than module order: Horizon's SR can load
        // a different number of kernels before FG starts.
        static const auto twins = [] {
            std::unordered_map<std::string, std::vector<unsigned char>> map;
            const char *folder = getenv("DLSSG_PROTOTYPE_REPLACEMENTS");
            if (!folder) return map;
            std::filesystem::path root=folder;
            if (!std::filesystem::exists(root)) return map;
            for (auto &entry : std::filesystem::directory_iterator(root)) {
                auto name = entry.path().filename().string();
                if (name.rfind("expected-module-", 0) != 0) continue;
                std::ifstream before(entry.path(), std::ios::binary);
                std::ifstream after(root/name.substr(9), std::ios::binary);
                std::string expected(std::istreambuf_iterator<char>(before), {});
                if (after) map.emplace(std::move(expected), std::vector<unsigned char>(std::istreambuf_iterator<char>(after), {}));
            }
            return map;
        }();
        auto found = twins.find(std::string(reinterpret_cast<const char *>(data), info->dataSize));
        if (found != twins.end()) changed = found->second;
        fprintf(stderr, "prototype cubin %u: input=%zu output=%zu ptx_edits=%zu magic=%02x%02x%02x%02x\n",
            serial, info->dataSize, changed.size(), replacements, data[0],data[1],data[2],data[3]);
    } else {
        fprintf(stderr, "prototype cubin %u: dump=%s size=%zu\n", serial, file, info->dataSize);
    }
    auto copied = *info;
    {
        std::lock_guard lock(*kept_mutex);
        // Repeated resize/backend switches reuse identical blobs, rather than
        // retaining another copy of every network kernel each time.
        auto [blob, inserted] = kept->emplace(reinterpret_cast<const char *>(changed.data()),changed.size());
        copied.pData = blob->data(); copied.dataSize = blob->size();
    }
    VkResult result = real_create(device, &copied, alloc, out);
    fprintf(stderr, "prototype cubin %u: vkCreateCuModuleNVX=%d\n", serial, result);
    if (result != VK_SUCCESS) failures++;
    return result;
}

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
    static auto actual = reinterpret_cast<PFN_vkGetDeviceProcAddr>(dlsym(dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL), "vkGetDeviceProcAddr"));
    auto function = actual(device, name);
    if (!strcmp(name, "vkCreateCuModuleNVX")) {
        real_create = reinterpret_cast<PFN_vkCreateCuModuleNVX>(function);
        fprintf(stderr, "prototype intercept vkCreateCuModuleNVX\n");
        return reinterpret_cast<PFN_vkVoidFunction>(create_module);
    }
    return function;
}
extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    static auto actual = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL), "vkGetInstanceProcAddr"));
    if (!strcmp(name, "vkGetDeviceProcAddr")) return reinterpret_cast<PFN_vkVoidFunction>(vkGetDeviceProcAddr);
    return actual(instance, name);
}
