#include <cstdlib>
#include <string>
#include <iostream>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"

#include "mediapipe/examples/desktop/mediapipe_face_mesh_api.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

#include "mediapipe/framework/deps/status.h"

// #include "mediapipe/framework/formats/rect.pb.h" // for bounding box.
#include "mediapipe/framework/formats/landmark.pb.h"  // for 

#define PRINT_LOG 0

// 对齐到MGSC_SDK
/// ai pixel format definition
typedef enum {
    AI_PIX_FMT_GRAY8 = 0,   ///< Y    1        8bpp ( 单通道8bit灰度像素 )
    //TODO support
    AI_PIX_FMT_YUV420P = 1, ///< YUV  4:2:0   12bpp ( 3通道, 一个亮度通道, 另两个为U分量和V分量通道, 所有通道都是连续的. 只支持I420)
    //TODO support
    AI_PIX_FMT_NV12 = 2,    ///< YUV  4:2:0   12bpp ( 2通道, 一个通道是连续的亮度通道, 另一通道为UV分量交错 )
    AI_PIX_FMT_NV21 = 3,    ///< YUV  4:2:0   12bpp ( 2通道, 一个通道是连续的亮度通道, 另一通道为VU分量交错 )
    AI_PIX_FMT_BGRA8888 = 4,///< BGRA 8:8:8:8 32bpp ( 4通道32bit BGRA 像素 )
    AI_PIX_FMT_BGR888 = 5,  ///< BGR  8:8:8   24bpp ( 3通道24bit BGR 像素 )
    AI_PIX_FMT_RGBA8888 = 6,///< RGBA 8:8:8:8 32bpp ( 4通道32bit RGBA 像素 )
    AI_PIX_FMT_RGB888 = 7   ///< RGB  8:8:8   24bpp ( 3通道24bit RGB 像素 )
} ai_pixel_format;


/// image rotate type definition
typedef enum: int{
    AI_CLOCKWISE_ROTATE_0 = 0,    ///< 图像不需要旋转,图像中的人脸为正脸
    AI_CLOCKWISE_ROTATE_90 = 90,  ///< 图像需要顺时针旋转90度,使图像中的人脸为正
    AI_CLOCKWISE_ROTATE_180 = 180,///< 图像需要顺时针旋转180度,使图像中的人脸为正
    AI_CLOCKWISE_ROTATE_270 = 270 ///< 图像需要顺时针旋转270度,使图像中的人脸为正
} ai_rotate_type;

// global variable
constexpr char kWindowName[] = "MediaPipe";
constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kOutputFaceLandmarks[] = "multi_face_landmarks";
constexpr char kOutputFaceBox[] = "face_rects_from_landmarks";

std::shared_ptr<mediapipe::CalculatorGraph> graph = nullptr;
std::unique_ptr<mediapipe::OutputStreamPoller> sop_facelandmark = nullptr;
std::unique_ptr<mediapipe::OutputStreamPoller> sop_facedetect = nullptr;
std::unique_ptr<mediapipe::OutputStreamPoller> sop_outputstream = nullptr;

static cv::TickMeter tm;

// TODO finish load tflite.
absl::Status _initMppGraph(std::string calculator_graph_config_contents)
{
    if (!graph)
    {
        graph = std::shared_ptr<mediapipe::CalculatorGraph>(new mediapipe::CalculatorGraph());
    }

    mediapipe::file::GetContents(calculator_graph_config_contents, &calculator_graph_config_contents);

    LOG(INFO) << "Get calculator graph config contents: "
        << calculator_graph_config_contents;
    mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);
    
    LOG(INFO) << "Initialize the calculator graph.";
    MP_RETURN_IF_ERROR(graph->Initialize(config));

    // 添加landmarks 输出流
    auto sop = graph->AddOutputStreamPoller(kOutputFaceLandmarks);
    assert(sop.ok());

    sop_facelandmark = std::make_unique<mediapipe::OutputStreamPoller>(std::move(sop.value()));
    
    // 添加facerect 输出流
    // mediapipe::StatusOrPoller sop_rect = graph.AddOutputStreamPoller(kOutputFaceBox);
    // sop_facedetect = std::make_unique<mediapipe::OutputStreamPoller>(std::move(sop_rect.value()));

    // 添加输出流
    // mediapipe::StatusOrPoller sop_out = graph.AddOutputStreamPoller(kOutputStream);
    // sop_outputstream = std::make_unique<mediapipe::OutputStreamPoller>(std::move(sop_out.value()));

	MP_RETURN_IF_ERROR(graph->StartRun({}));
    return absl::OkStatus();
}

int initMppGraph(std::string calculator_graph_config_contents)
{
    absl::Status s = _initMppGraph(calculator_graph_config_contents);

#if PRINT_LOG
    LOG(INFO) << "Log in initMppGraph: " << s.message();
#endif

    if (s.ok())
    {
        return 1;
    }
    else 
    {
        return 0;
    }
}

absl::Status _releaseMppGraph()
{
    MP_RETURN_IF_ERROR(graph->CloseInputStream(kInputStream));
    return graph->WaitUntilDone();
}

int releaseMppGraph()
{
    sop_facelandmark.release();

    auto s = _releaseMppGraph();
    graph.reset();

#if PRINT_LOG
        LOG(INFO) << "Log in releaseMppGraph: " << s.message();
#endif
    if (s.ok())
    {
        return 1;
    }        
    else
    {
         return 0;
    }
}

absl::Status _runMppGraph(char* buffer, ai_pixel_format pixel_format, const int width, const int height, ai_rotate_type rotate, bool flip, std::vector<std::vector<float> >& result)
{
    // graph.WaitUntilIdle();
    // Create cvMat
    // tm.reset();
    // tm.start();
    cv::Mat img;

    if(AI_PIX_FMT_RGB888 == pixel_format)
    {
        img = cv::Mat(height, width, CV_8UC3, buffer);
        img.copyTo(img);
    } 
    else if(AI_PIX_FMT_BGR888 == pixel_format)
    {
        cv::Mat img_bgr(height, width, CV_8UC3, buffer);
        cv::cvtColor(img_bgr, img, cv::COLOR_BGR2RGB);
    } 
    else if(AI_PIX_FMT_RGBA8888 == pixel_format)
    {
        cv::Mat img_rgba(height, width, CV_8UC4, buffer);
        cv::cvtColor(img_rgba, img, cv::COLOR_RGBA2RGB);
    }
    else if(AI_PIX_FMT_BGRA8888 == pixel_format)
    {
        cv::Mat img_bgra(height, width, CV_8UC4, buffer);
        cv::cvtColor(img_bgra, img, cv::COLOR_BGRA2RGB);
    }
    else
    {
        return absl::AbortedError("runMedipipeFaceMesh error! Unsupported Image type!");
    }

    if (AI_CLOCKWISE_ROTATE_0 != rotate) 
    {
        int rotateTarget;
        if (rotate == AI_CLOCKWISE_ROTATE_270) {
            rotateTarget = cv::ROTATE_90_COUNTERCLOCKWISE;
        }
        else if (rotate == AI_CLOCKWISE_ROTATE_180) {
            rotateTarget = cv::ROTATE_180;
        }
        else if (rotate == AI_CLOCKWISE_ROTATE_90) {
            rotateTarget = cv::ROTATE_90_CLOCKWISE;
        }
        else 
        {
            return absl::AbortedError("Unsupported rotate type! only 0, 90, 180, 270 are supported!");
        }
        //  前置摄像头，需要先旋转再水平镜像
        //cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::rotate(img, img, rotateTarget);
    }

    if (flip) 
    {
        cv::flip(img, img, 1);
    }

    // cv::imshow("r datma", img);
    // cv::waitKey(0);

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, img.cols, img.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    img.copyTo(mediapipe::formats::MatView(input_frame.get()));

    // Add the output node to graph.
    // if (showImg && !sop_outputstream)
    // {
    //     // graph.WaitUntilDone(); 
    //     mediapipe::StatusOrPoller sop_out = graph.AddOutputStreamPoller(kOutputStream);
    //     sop_outputstream = std::make_unique<mediapipe::OutputStreamPoller>(std::move(sop_out.value()));

    //     MP_RETURN_IF_ERROR(graph.StartRun({}));
    // }

    // Send image packet into graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

    // tm.stop();
    // LOG(INFO)<<"time preprocess = "<<tm.getTimeMilli();

    // tm.reset();
    // tm.start();
    MP_RETURN_IF_ERROR(graph->AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));

    // LOG(INFO)<<"the input timestamp_us = "<<frame_timestamp_us;
    graph->WaitUntilIdle(); 

    // tm.stop();
    // LOG(INFO)<<"time inference = "<<tm.getTimeMilli();
    // if (showImg)
    // {

    //     mediapipe::Packet packet;
    //     if (sop_outputstream->QueueSize() == 0 || !sop_outputstream->Next(&packet)) 
    //     {
    //         // LOG(INFO)<<"can not get, return!";
    //         return absl::OkStatus();
    //     }
    //     // else
    //     // {
    //     //     LOG(INFO)<<"get output pack";
    //     // }

    //     if (showImg)
    //     {
    //         // LOG(INFO)<<"gout Output!";
    //         auto& output_frame = packet.Get<mediapipe::ImageFrame>();
    //         cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
    //         cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    //         cv::imshow(kWindowName, output_frame_mat);

    //         const int pressed_key = cv::waitKey(5);
    //     }
    // }
    
    // else
    // {
        // LOG(INFO)<<"do not show img!";
        // auto& output_frame = packet.Get<mediapipe::ImageFrame>();
        // cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        // cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    // }

    // process face rect.
    // mediapipe::Packet packet_faceBox;
    // if (sop_facedetect->QueueSize() > 0 && sop_facedetect->Next(&packet_faceBox))
    // {
    //     // LOG(INFO)<<"gout Rect!, queue size = "<<sop_facedetect->QueueSize()<<" the output timestamp = "<<packet_faceBox.Timestamp().Value();
    //     std::vector<mediapipe::NormalizedRect> output_rect = packet_faceBox.Get<std::vector<mediapipe::NormalizedRect>>();
        
    //     assert(output_rect.size() == 1);
        
    //     result->face_rect.x_center = output_rect[0].x_center() * img.cols;
    //     result->face_rect.y_center = output_rect[0].y_center() * img.rows;
    //     result->face_rect.w = output_rect[0].width() *  img.cols;
    //     result->face_rect.h = output_rect[0].height() *  img.rows;
    // }
    // else
    // {
    //     // if (sop_facedetect->QueueSize() == 0)
    //     // {
    //     //     std::cout<<"No Rect!"<<std::endl;
    //     // } 
    //     // else 
    //     // if (sop_facedetect->Next(&packet_faceBox))
    //     // {
    //     //     LOG(INFO)<<"Can not got sop face Box!";
    //     // }
    // }
    result.clear();
    mediapipe::Packet packet_landmarks;
    if (sop_facelandmark->QueueSize() != 0 && sop_facelandmark->Next(&packet_landmarks)) 
    {
        // LOG(INFO)<<"gout Landmark!";
        std::vector<mediapipe::NormalizedLandmarkList> output_landmarks = packet_landmarks.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

        // LOG(INFO) << "output_landmarks size = "<<output_landmarks.size()<<", size0 = "<<output_landmarks[0].landmark_size();
        
        int face_num = output_landmarks.size();
        result.resize(face_num);
        int lnum = output_landmarks[0].landmark_size();

        for (int k = 0; k < face_num; k++)
        {
            result[k].clear();
            result[k].resize(FACE_MESH_NUM*3, 0.f);
            for (int i = 0; i < lnum; i++)
            {
                const mediapipe::NormalizedLandmark landmark = output_landmarks[k].landmark(i);
                result[k][i*3]     = landmark.x();
                result[k][i*3 + 1] = landmark.y();
                result[k][i*3 + 2] = landmark.z();

                // LOG(INFO) <<i<<", x = "<<landmark.x() * img.cols<<", y = "<<landmark.y() * img.rows
                //     <<", z = "<<landmark.z();
            }
        }
    }
    #if PRINT_LOG
    else
    {
        LOG(INFO)<<"Can not got land mark face Box!";
    }
    #endif
    // else
    // {
    //     LOG(INFO)<<"Can not got land mark face Box!";
    // }
    // graph.WaitUntilIdle();
    return absl::OkStatus();
}


int runMppGraph(char* buffer, const int pixel_format, const int width, const int height, int rotate, bool flip, std::vector<std::vector<float> >& result)
{
    auto s = _runMppGraph(buffer, (ai_pixel_format)pixel_format, width, height, (ai_rotate_type)rotate, flip, result);
#if PRINT_LOG
        LOG(INFO) << "Log in runMppGraphDirct: " << s.message();
#endif

    if (s.ok())
    {
        return 1;
    }
    else
    {
        return 0;
    }
}