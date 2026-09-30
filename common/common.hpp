#ifndef COMMON_HPP
#define COMMON_HPP
#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t _err = (call);                                           \
        if (_err != hipSuccess) {                                           \
            std::fprintf(stderr, "HIP error %s at %s:%d: %s\n",             \
                         #call, __FILE__, __LINE__, hipGetErrorString(_err)); \
            std::abort();                                                   \
        }                                                                   \
    } while (0)


inline constexpr int WarpSize = 32; // RDNA2 Wave32
inline constexpr int kQuantBlockSize = 32;
#endif // COMMON_HPP