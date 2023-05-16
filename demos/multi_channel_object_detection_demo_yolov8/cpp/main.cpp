// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
/**
 * \brief The entry point for the OpenVINIO multichannel_yolo_detection demo application
 * \file multichannel_yolo_detection/main.cpp
 * \example multichannel_yolo_detection/main.cpp
 */
#include <iostream>
#include <vector>
#include <utility>

#include <algorithm>
#include <mutex>
#include <queue>
#include <chrono>
#include <sstream>
#include <memory>
#include <string>
#include <random>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#endif

#include <opencv2/opencv.hpp>
#include <openvino/op/region_yolo.hpp>

#include <monitors/presenter.h>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

#include "input.hpp"
#include "multichannel_params.hpp"
#include "output.hpp"
#include "threading.hpp"
#include "graph.hpp"
#include "yolov8.hpp"

// #define DEBUG

namespace
{
    constexpr char threshold_message[] = "Probability threshold for detections";
    DEFINE_double(t, 0.5, threshold_message);
    
    // Display Settings.
    const size_t DISP_WIDTH = 960;
    const size_t DISP_HEIGHT = 540;
    const size_t MAX_INPUTS = 25;

    void parse(int argc, char *argv[])
    {
        gflags::ParseCommandLineFlags(&argc, &argv, false);
        slog::info << ov::get_openvino_version() << slog::endl;
        if (FLAGS_h || argc == 1)
        {
            std::cout << "\n    [-h]              " << help_message
                      << "\n     -i               " << input_message
                      << "\n    [-loop]           " << loop_message
                      << "\n    [-duplicate_num]  " << duplication_channel_number_message
                      << "\n     -m <path>        " << model_path_message
                      << "\n    [-d <device>]     " << target_device_message
                      << "\n    [-n_iqs]          " << input_queue_size
                      << "\n    [-fps_sp]         " << fps_sampling_period
                      << "\n    [-n_sp]           " << num_sampling_periods
                      << "\n    [-t]              " << threshold_message
                      << "\n    [-no_show]        " << no_show_message
                      << "\n    [-show_stats]     " << show_statistics
                      << "\n    [-real_input_fps] " << real_input_fps
                      << "\n    [-u]              " << utilization_monitors_message << '\n';
            showAvailableDevices();
            std::exit(0);
        }
        if (FLAGS_m.empty())
        {
            throw std::runtime_error("Parameter -m is not set");
        }
        if (FLAGS_i.empty())
        {
            throw std::runtime_error("Parameter -i is not set");
        }
        if (FLAGS_duplicate_num == 0)
        {
            throw std::runtime_error("Parameter -duplicate_num must be positive");
        }
        if (FLAGS_bs != 1)
        {
            throw std::runtime_error("Parameter -bs must be 1");
        }
    }

    void printInputAndOutputsInfo(const ov::Model &network)
    {
        slog::info << "model name: " << network.get_friendly_name() << slog::endl;

        const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
        for (const ov::Output<const ov::Node> input : inputs)
        {
            slog::info << "    inputs" << slog::endl;

            const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
            slog::info << "        input name: " << name << slog::endl;

            const ov::element::Type type = input.get_element_type();
            slog::info << "        input type: " << type << slog::endl;

            const ov::Shape shape = input.get_shape();
            slog::info << "        input shape: " << shape << slog::endl;
        }

        const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
        for (const ov::Output<const ov::Node> output : outputs)
        {
            slog::info << "    outputs" << slog::endl;

            const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
            slog::info << "        output name: " << name << slog::endl;

            const ov::element::Type type = output.get_element_type();
            slog::info << "        output type: " << type << slog::endl;

            const ov::Shape shape = output.get_shape();
            slog::info << "        output shape: " << shape << slog::endl;
        }
    }

    struct DetectionObject
    {
        int xmin, ymin, width, height, class_id;
        float confidence;
        cv::Scalar color;
        std::string class_name;
        std::vector<cv::Point> segment;

        DetectionObject(int x1, int y1, int x2, int y2, int class_id, float confidence,
                        std::string class_name, cv::Scalar color, std::vector<cv::Point> segment) : xmin{x1},
                                                                                                    ymin{y1},
                                                                                                    width{x2},
                                                                                                    height{y2},
                                                                                                    class_id{class_id},
                                                                                                    confidence{confidence},
                                                                                                    class_name{class_name},
                                                                                                    color{color},
                                                                                                    segment{segment}
        {
        }

        bool operator<(const DetectionObject &s2) const
        {
            return this->confidence < s2.confidence;
        }
        bool operator>(const DetectionObject &s2) const
        {
            return this->confidence > s2.confidence;
        }
    };

    struct DisplayParams
    {
        std::string name;
        cv::Size windowSize;
        cv::Size frameSize;
        size_t count;
        cv::Point points[MAX_INPUTS];
    };

    DisplayParams prepareDisplayParams(size_t count)
    {
        DisplayParams params;
        params.name = "YOLOv8";
        params.count = count;
        params.windowSize = cv::Size(DISP_WIDTH, DISP_HEIGHT);

        size_t gridCount = static_cast<size_t>(ceil(sqrt(count)));
        size_t gridStepX = static_cast<size_t>(DISP_WIDTH / gridCount);
        size_t gridStepY = static_cast<size_t>(DISP_HEIGHT / gridCount);
        if (gridStepX == 0 || gridStepY == 0)
        {
            throw std::logic_error("Can't display every input: there are too many of them");
        }
        params.frameSize = cv::Size(gridStepX, gridStepY);

        for (size_t i = 0; i < count; i++)
        {
            cv::Point p;
            p.x = gridStepX * (i / gridCount);
            p.y = gridStepY * (i % gridCount);
            params.points[i] = p;
        }
        return params;
    }

    void drawDetections(cv::Mat &img, const std::vector<DetectionObject> &detections)
    {
        // printf("drawDetections\n");
        for (const DetectionObject &f : detections)
        {
            // bounding box
            int tl = round(0.001 * (img.size[0] + img.size[1]) / 2) + 1;
            cv::rectangle(img,
                          cv::Rect2f(static_cast<float>(f.xmin),
                                     static_cast<float>(f.ymin),
                                     static_cast<float>(f.width),
                                     static_cast<float>(f.height)),
                          f.color,
                          tl);

            std::string label = "";
            if (f.class_id != NAN && f.confidence != NAN)
            {
                std::ostringstream oss;
                oss << yolov8::CLASS_NAMES[f.class_id] << " " << f.confidence;
                label = oss.str();
            }

            // text
            if (label != "")
            {
                cv::putText(img, label, cv::Point2f(f.xmin, f.ymin - 2), 0, MAX(tl / 3, 1), cv::Scalar(255, 255, 255), MAX(tl - 1, 1));
            }

            // segmentations
            if (f.segment.size() > 0)
            {
                cv::Mat img_with_mask;
                img.copyTo(img_with_mask);
                cv::fillPoly(img_with_mask, std::vector<std::vector<cv::Point>>{f.segment}, f.color);
                cv::addWeighted(img, 0.5, img_with_mask, 0.5, 1, img);
            }
        }
    }

    void displayNSources(const std::vector<std::shared_ptr<VideoFrame>> &data,
                         const std::string &stats,
                         const DisplayParams &params,
                         Presenter &presenter,
                         PerformanceMetrics &metrics,
                         bool no_show)
    {
        cv::Mat windowImage = cv::Mat::zeros(params.windowSize, CV_8UC3);
        auto loopBody = [&](size_t i)
        {
            // Draw Detections
            auto &elem = data[i];
            if (!elem->frame.empty())
            {
                cv::Rect rectFrame = cv::Rect(params.points[i], params.frameSize);
                cv::Mat windowPart = windowImage(rectFrame);
                cv::Mat result;
                elem->frame.copyTo(result);

                drawDetections(result, elem->detections.get<std::vector<DetectionObject>>());

                // resize to display shape after drawing detections
                cv::resize(result, windowPart, params.frameSize);
            }
        };

//  #ifdef USE_TBB
#if 0 // disable multithreaded rendering for now
    run_in_arena([&](){
        tbb::parallel_for<size_t>(0, data.size(), [&](size_t i) {
            loopBody(i);
        });
    });
#else
        for (size_t i = 0; i < data.size(); ++i)
        {
            loopBody(i);
        }
#endif
        presenter.drawGraphs(windowImage);

        for (size_t i = 0; i < data.size() - 1; ++i)
        {
            metrics.update(data[i]->timestamp);
        }
        metrics.update(data.back()->timestamp, windowImage, {10, 22}, cv::FONT_HERSHEY_COMPLEX, 0.65);
        if (!no_show)
        {
            cv::imshow(params.name, windowImage);
        }
    }

} // namespace

int main(int argc, char *argv[])
{
    try
    {
#if USE_TBB
        TbbArenaWrapper arena;
#endif
        parse(argc, argv);
        const std::vector<std::string> &inputs = split(FLAGS_i, ',');
        DisplayParams params = prepareDisplayParams(inputs.size() * FLAGS_duplicate_num);

        // PrePostProcess
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(FLAGS_m);
        if (model->get_parameters().size() != 1)
        {
            throw std::logic_error("Face Detection model must have only one input");
        }
        ov::Shape input_shape = {1, 640, 640, 3};
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input().tensor().set_shape(input_shape).set_element_type(ov::element::u8).set_layout("NHWC");
        ppp.input().preprocess().convert_element_type(ov::element::f32).scale(255.f).convert_layout("NCHW");
        ppp.input().model().set_layout("NCHW");
        for (const ov::Output<ov::Node> &out : model->outputs())
        {
            ppp.output(out.get_any_name()).tensor().set_element_type(ov::element::f32);
        }
        model = ppp.build();
        ov::set_batch(model, FLAGS_bs);
        printInputAndOutputsInfo(*model);

        // Detection Color
        yolov8::ColorPlatte colorPlatte = yolov8::ColorPlatte();

        std::queue<ov::InferRequest> reqQueue = compile(std::move(model),
                                                        FLAGS_m, FLAGS_d, 2 /*roundUp(params.count, FLAGS_bs)*/, core);
        // get model input shape
        ov::Shape inputShape = reqQueue.front().get_input_tensor().get_shape();
        if (4 != inputShape.size())
        {
            throw std::runtime_error("Invalid model input dimensions");
        }

#ifdef DEBUG
        std::cout << "REQ INPUT SHAPE =" << inputShape << std::endl;
#endif
        IEGraph graph{std::move(reqQueue), FLAGS_show_stats};

        VideoSources::InitParams vsParams;
        vsParams.inputs = inputs;
        vsParams.loop = FLAGS_loop;
        vsParams.queueSize = FLAGS_n_iqs;
        vsParams.collectStats = FLAGS_show_stats;
        vsParams.realFps = FLAGS_real_input_fps;
        vsParams.expectedHeight = static_cast<unsigned>(inputShape[2]);
        vsParams.expectedWidth = static_cast<unsigned>(inputShape[3]);

        VideoSources sources(vsParams);
        sources.start();

        size_t currentFrame = 0;
        // original frame shape from input video
        int VIDEO_HEIGHT = 0;
        int VIDEO_WIDTH = 0;
        
        graph.start(
            FLAGS_bs, [&](VideoFrame &img)
            {
                img.sourceIdx = currentFrame;
                size_t camIdx = currentFrame / FLAGS_duplicate_num;
                currentFrame = (currentFrame + 1) % (sources.numberOfInputs() * FLAGS_duplicate_num);
                bool got = sources.getFrame(camIdx, img);

                VIDEO_HEIGHT = img.frame.size[0];
                VIDEO_WIDTH = img.frame.size[1];

                return got; },
            [&](ov::InferRequest req,
                cv::Size frameSize)
            {
                /*
                YOLOv8 model postprocessing function.
                */

                std::vector<DetectionObject> objects;

                // MODEL OUTPUT
                const ov::Tensor &box_tensor = req.get_output_tensor(0);
                ov::Tensor mask_tensor = req.get_output_tensor(1);
                auto box_shape = box_tensor.get_shape();
                auto mask_shape = mask_tensor.get_shape();

                // SHAPE
                static const int INPUT_HEIGHT = inputShape[1];
                static const int INPUT_WIDTH = inputShape[2];

                // SEGMENTATION
                static const int SEG_WIDTH = mask_shape[3];    // 160
                static const int SEG_HEIGHT = mask_shape[2];   // 160
                static const int SEG_CHANNELS = mask_shape[1]; // 32
                // BOUNDING BOX
                static const int CLASSES = 80;
                int NET_LENGHT_OUT0 = box_shape[1];      // 4 + CLASSES + SEG_CHANNELS = 116
                static const int NUM_BOX = box_shape[2]; // 8400
                static const float CONF_THRESH = 0.25;
                static const float IOU_THRESH = 0.7;

                // TEMP RESULTS
                std::vector<int> class_ids;
                std::vector<float> confidences;
                std::vector<cv::Rect> boxes_proto;
                std::vector<cv::Mat> masks_proto;

                // out0 -> box: 116 * 8400
                cv::Mat out0 = cv::Mat(NET_LENGHT_OUT0, NUM_BOX, CV_32F, box_tensor.data());
                // out1 -> mask: 32 * (160*160), viewed here
                cv::Mat out1 = cv::Mat(SEG_CHANNELS, (SEG_HEIGHT * SEG_WIDTH), CV_32F, mask_tensor.data());

#ifdef DEBUG
                std::cout << "out0: " << out0.size << " out1: " << out1.size << "\n";
                std::cout << "inputhw [" << INPUT_HEIGHT << "," << INPUT_WIDTH << "]\n";
                std::cout << "shape [" << VIDEO_HEIGHT << "," << VIDEO_WIDTH << "]\n";
#endif
                // gain = model input shape / input video shape
                float gain = MIN((float)INPUT_HEIGHT / VIDEO_HEIGHT, (float)INPUT_WIDTH / VIDEO_WIDTH);
                // wh padding
                float pad[2] = {(float)(INPUT_WIDTH - VIDEO_WIDTH * gain) / 2, (float)(INPUT_HEIGHT - VIDEO_HEIGHT * gain) / 2};

#ifdef DEBUG
                std::cout << "paddings (" << pad[0] << "," << pad[1] << ")\n";
                std::cout << "gain =" << gain << std::endl;
#endif

                // Filter Box by CONF_THRESH
                for (int i = 0; i < NUM_BOX; i++)
                {
                    // i: witch box, 4: start at first score (x,y,w,h, box1 score, box2 score,...), 1: only take this box, CLASSES: get all class score
                    cv::Mat scores = out0(cv::Rect(i, 4, 1, CLASSES)).clone();
                    cv::Point classIdPoint;
                    double max_class_score;

                    minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);

                    max_class_score = (float)max_class_score;
                    if (max_class_score > CONF_THRESH)
                    {
                        float x = out0.at<float>(0, i);
                        float y = out0.at<float>(1, i);
                        float w = out0.at<float>(2, i);
                        float h = out0.at<float>(3, i);

                        class_ids.push_back(classIdPoint.y);
                        confidences.push_back(max_class_score);
                        boxes_proto.push_back(cv::Rect(x, y, w, h));

                        // Save Mask.
                        cv::Mat temp_mask = out0(cv::Rect(i, 4 + CLASSES, 1, SEG_CHANNELS)).clone(); // mask that relative to the box.
                        masks_proto.push_back(temp_mask.t());
                    }
                }

                std::vector<int> nms_result;
                cv::dnn::NMSBoxes(boxes_proto, confidences, CONF_THRESH, IOU_THRESH, nms_result);

                int inputhw[2] = {INPUT_HEIGHT, INPUT_WIDTH};

                for (unsigned long i = 0; i < nms_result.size(); ++i)
                {
                    int idx = nms_result[i];

                    /*
                    Revert letterbox operation to get real box pos (in the format of xywh).
                    XYXY fromat should be:
                        boxes[..., [0, 2]] -= pad[0]  # x padding
                        boxes[..., [1, 3]] -= pad[1]  # y padding
                        boxes[..., :4] /= gain
                    */

                    auto box = boxes_proto[idx];
                    int left = (box.x - pad[0] - 0.5 * box.width) / gain;
                    int top = (box.y - pad[1] - 0.5 * box.height) / gain;
                    int width = (box.width) / gain;
                    int height = (box.height) / gain;

                    // detection box, with xywh2xyxy
                    float rectt[4] = {box.x - box.width / 2, box.y - box.height / 2, box.x + box.width / 2, box.y + box.height / 2};

                    // segmentation mask
                    auto mask = yolov8::process_mask(out1, masks_proto[idx], cv::Mat(1, 4, CV_32F, rectt), inputhw, mask_shape);
                    std::vector<cv::Point> segment = yolov8::scale_segment(gain, pad, yolov8::mask2segment(mask), VIDEO_HEIGHT, VIDEO_WIDTH);
                    // std::vector<cv::Point> segment = mask2segment(mask);

                    int class_id = class_ids[idx];
                    DetectionObject obj(left, top, width, height, class_ids[idx], confidences[idx], yolov8::CLASS_NAMES[class_id], colorPlatte.getColorsByID(class_id), segment);
                    objects.push_back(obj);
                }

                // Final detection object
                std::vector<Detections> detections(1);
                detections[0].set(new std::vector<DetectionObject>);

                for (auto &object : objects)
                {
                    detections[0].get<std::vector<DetectionObject>>().push_back(object);
                }

                return detections;
            });

        std::mutex statMutex;
        std::stringstream statStream;

        cv::Size graphSize{static_cast<int>(params.windowSize.width / 4), 60};
        Presenter presenter(FLAGS_u, params.windowSize.height - graphSize.height - 10, graphSize);
        PerformanceMetrics metrics;

        const size_t outputQueueSize = 1;

        if (!FLAGS_no_show)
        {
            cv::namedWindow(params.name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO | cv::WINDOW_GUI_EXPANDED);
        }

        AsyncOutput output(FLAGS_show_stats, outputQueueSize,
                           [&](const std::vector<std::shared_ptr<VideoFrame>> &result)
                           {
                               std::string str;
                               if (FLAGS_show_stats)
                               {
                                   std::unique_lock<std::mutex> lock(statMutex);
                                   str = statStream.str();
                               }
                               displayNSources(result, str, params, presenter, metrics, FLAGS_no_show);
                               int key = cv::waitKey(1);
                               presenter.handleKey(key);

                               return (key != 27);
                           });

        output.start();

        std::vector<std::shared_ptr<VideoFrame>> batchRes;
        using timer = std::chrono::high_resolution_clock;
        using duration = std::chrono::duration<float, std::milli>;
        timer::time_point lastTime = timer::now();
        duration samplingTimeout(FLAGS_fps_sp);

        size_t perfItersCounter = 0;

        while (sources.isRunning() || graph.isRunning())
        {
            bool readData = true;
            while (readData)
            {
                auto br = graph.getBatchData(params.frameSize);
                if (br.empty())
                {
                    break; // IEGraph::getBatchData had nothing to process and returned. That means it was stopped
                }
                for (size_t i = 0; i < br.size(); i++)
                {
                    // this approach waits for the next input image for sourceIdx. If provided a single image,
                    // it may not show results, especially if -real_input_fps is enabled
                    auto val = static_cast<unsigned int>(br[i]->sourceIdx);
                    auto it = find_if(batchRes.begin(), batchRes.end(), [val](const std::shared_ptr<VideoFrame> &vf)
                                      { return vf->sourceIdx == val; });
                    if (it != batchRes.end())
                    {
                        output.push(std::move(batchRes));
                        batchRes.clear();
                        readData = false;
                    }
                    batchRes.push_back(std::move(br[i]));
                }
            }

            if (!output.isAlive())
            {
                break;
            }

            auto currTime = timer::now();
            auto deltaTime = (currTime - lastTime);
            if (deltaTime >= samplingTimeout)
            {
                lastTime = currTime;

                if (FLAGS_show_stats)
                {
                    if (++perfItersCounter >= FLAGS_n_sp)
                    {
                        break;
                    }
                }

                if (FLAGS_show_stats)
                {
                    std::unique_lock<std::mutex> lock(statMutex);
                    slog::debug << "------------------- Frame # " << perfItersCounter << "------------------" << slog::endl;
                    writeStats(slog::debug, slog::endl, sources.getStats(), graph.getStats(), output.getStats());
                    statStream.str(std::string());
                    writeStats(statStream, '\n', sources.getStats(), graph.getStats(), output.getStats());
                }
            }
        }
        slog::info << "Metrics report:" << slog::endl;
        metrics.logTotal();
        slog::info << presenter.reportMeans() << slog::endl;
    }
    catch (const std::exception &error)
    {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...)
    {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }
    return 0;
}
