#ifndef MPP_INTERFACE_H_
#define MPP_INTERFACE_H_

#include <iostream>
#include <vector>

#if defined (_MSC_VER) || defined (_WIN32)
#ifndef MG_EXPORTS
#define  MG_EXPORTS __declspec(dllexport)
#else
#define  MG_EXPORTS __declspec(dllimport)
#endif // MG_EXPORTS
#else // defined (windows)
#define MG_EXPORTS
#endif

#define FACE_MESH_NUM 478 // Face point is 3D.

/// @brief init the mediapipe graph.
/// @param model_path_folder the mediapipe graph file. with suffix .
MG_EXPORTS int initMppGraph(std::string model_path_folder);

/// @brief  run mediapipe with given Face image
/// @param buffer image pointer with data insid.
/// @param image_fromat 对齐MGSC SDK，只有BGR，RGB，BGRA，RGBA四种格式支持。
/// @param width image width
/// @param height image height
/// @param rotate 顺时针旋转角度，仅支持0，90，180，270
/// @param flip 是否翻转。
/// @param result two dimension array [N, 478 x 3] (N means N face has been found) of std::vector<std::vector<float > >.
/// @return 
MG_EXPORTS int runMppGraph(char* buffer, const int image_fromat, const int width, const int height, int rotate, bool flip, std::vector<std::vector<float> >& result);

// typedef enum {
//     AI_PIX_FMT_GRAY8 = 0,   ///< Y    1        8bpp ( 单通道8bit灰度像素 )
//     //TODO support
//     AI_PIX_FMT_YUV420P = 1, ///< YUV  4:2:0   12bpp ( 3通道, 一个亮度通道, 另两个为U分量和V分量通道, 所有通道都是连续的. 只支持I420)
//     //TODO support
//     AI_PIX_FMT_NV12 = 2,    ///< YUV  4:2:0   12bpp ( 2通道, 一个通道是连续的亮度通道, 另一通道为UV分量交错 )
//     AI_PIX_FMT_NV21 = 3,    ///< YUV  4:2:0   12bpp ( 2通道, 一个通道是连续的亮度通道, 另一通道为VU分量交错 )
//     AI_PIX_FMT_BGRA8888 = 4,///< BGRA 8:8:8:8 32bpp ( 4通道32bit BGRA 像素 )
//     AI_PIX_FMT_BGR888 = 5,  ///< BGR  8:8:8   24bpp ( 3通道24bit BGR 像素 )
//     AI_PIX_FMT_RGBA8888 = 6,///< RGBA 8:8:8:8 32bpp ( 4通道32bit RGBA 像素 )
//     AI_PIX_FMT_RGB888 = 7   ///< RGB  8:8:8   24bpp ( 3通道24bit RGB 像素 )
// } ai_pixel_format;

/// @brief release the resource of Mediapipe.
MG_EXPORTS int releaseMppGraph();

#endif
