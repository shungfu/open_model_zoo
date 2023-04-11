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

// #define DEBUG

namespace
{
    constexpr char threshold_message[] = "Probability threshold for detections";
    DEFINE_double(t, 0.5, threshold_message);

    const std::vector<std::string> CLASS_NAMES = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};

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
        int xmin, ymin, xmax, ymax, class_id;
        float confidence;
        cv::Scalar color;
        std::string class_name;

        DetectionObject(double x1, double y1, double x2, double y2, int class_id, float confidence, std::string class_name, cv::Scalar color) : xmin{x1},
                                                                                                                                                ymin{y1},
                                                                                                                                                xmax{x2},
                                                                                                                                                ymax{y2},
                                                                                                                                                class_id{class_id},
                                                                                                                                                confidence{confidence},
                                                                                                                                                class_name{class_name},
                                                                                                                                                color{color}
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

    struct Detection
    {
        int class_id{0};
        std::string class_name{};
        float confidence{0.0};
        cv::Scalar color{};
        cv::Rect box{};
    };

    const size_t DISP_WIDTH = 1920;
    const size_t DISP_HEIGHT = 1080;
    const size_t MAX_INPUTS = 25;

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
            // cv::Scalar color = cv::Scalar(255, 0, 0);
            int tl = round(0.001 * (img.size[0] + img.size[1]) / 2) + 1;
            cv::rectangle(img,
                          cv::Rect2f(static_cast<float>(f.xmin),
                                     static_cast<float>(f.ymin),
                                     static_cast<float>((f.xmax)),
                                     static_cast<float>((f.ymax))),
                          f.color,
                          tl);

            std::string label = "";
            if (f.class_id != NAN && f.confidence != NAN)
            {
                std::ostringstream oss;
                oss << CLASS_NAMES[f.class_id] << " " << f.confidence;
                label = oss.str();
            }

            // text
            if (label != "")
            {
                cv::putText(img, label, cv::Point2f(f.xmin, f.ymin - 2), 0, MAX(tl / 3, 1), cv::Scalar(255, 255, 255), MAX(tl - 1, 1));
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
            auto &elem = data[i];
            if (!elem->frame.empty())
            {
                cv::Rect rectFrame = cv::Rect(params.points[i], params.frameSize);
                cv::Mat windowPart = windowImage(rectFrame);
                cv::Mat result;
                elem->frame.copyTo(result);

                drawDetections(result, elem->detections.get<std::vector<DetectionObject>>());
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

    struct ColorPlatte
    {
        // Ultralytics color palette: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/utils/plotting.py#L25
    private:
        std::vector<std::string> hexs{"FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86", "1A9334", "00D4BB",
                                      "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF", "520085", "CB38FF", "FF95C8", "FF37C7"};
        std::vector<cv::Scalar> palette;

    public:
        cv::Scalar getColorsByID(int i, bool use_bgr = false)
        {
            cv::Scalar rgb = this->palette[i % this->palette.size()];

            if (use_bgr)
            {
                return cv::Scalar(rgb[2], rgb[1], rgb[0]);
            }
            return rgb;
        }

        void hex2rgb(std::string h)
        {
            unsigned int rgb[3] = {0, 0, 0};
            int start_pos = 0;
            for (int i = 0; i < 3; i++)
            {
                std::string substr = "0x" + h.substr(start_pos, 2);
                start_pos += 2;
                rgb[i] = std::stoul(substr, nullptr, 16);
            }
            this->palette.push_back(cv::Scalar(rgb[0], rgb[1], rgb[2]));
        }

        ColorPlatte()
        {
            // hex2 rgb
            for (int i = 0; i < hexs.size(); i++)
            {
                hex2rgb(hexs[i]);
            }
        }
    };

    void crop_mask(cv::Mat masks, cv::Mat boxes)
    {
        /*
        TODO:
        source: https://github.com/ultralytics/ultralytics/blob/fe61018975182f4d7645681b4ecc09266939dbfb/ultralytics/yolo/utils/ops.py#L538

        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box
        Args:
            masks (torch.Tensor): [h, w, n] tensor of masks
            boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form
        Returns:
            (torch.Tensor): The masks are being cropped to the bounding box.
        */
        int n = masks.size[0], h = masks.size[1], w = masks.size[2];
        std::vector<float> x1 = boxes.col(0), y1 = boxes.col(1), x2 = boxes.col(2), y2 = boxes.col(3);
        
        // for (auto const &value : x1)
        // {
        //     std::cout << value << " ";
        // }
        // std::cout << std::endl;


    }

    void process_mask(cv::Mat protos, cv::Mat mask_in, cv::Mat bboxes, int inputhw[2], bool upsample = true)
    {
        /*
        source: https://github.com/ultralytics/ultralytics/blob/fe61018975182f4d7645681b4ecc09266939dbfb/ultralytics/yolo/utils/ops.py#L578

        It takes the output of the mask head, and applies the mask to the bounding boxes. This is faster but produces
        downsampled quality of mask

        Args:
            protos (torch.Tensor): [mask_dim, mask_h, mask_w]
            masks_in (torch.Tensor): [n, mask_dim], n is number of masks after nms
            bboxes (torch.Tensor): [n, 4], n is number of masks after nms
            shape (tuple): the size of the input image (h,w)

        Returns:
            (torch.Tensor): The processed masks.
        */

        int c = protos.size[0], mh = protos.size[1], mw = protos.size[2]; // CHW
        int ih = inputhw[0], iw = inputhw[1];

        std::vector<int> dims{protos.size[0], mh * mw};
        cv::Mat viewed = cv::Mat(dims.size(), dims.data(), protos.type(), protos.data); // view(c,-1)

        // std::cout << "viewed: " << viewed.at<float>(0, 160) << "\n";
        // std::cout << "protos(0): " << protos.at<float>(0, 0, 159) << "\n";
        // std::cout << "protos(1): " << protos.at<float>(0, 1, 0) << "\n";
        // std::cout << "maskin: " << mask_in.size << "\n";
        // std::cout << "maskin * viewed = " << mask_in.size << " " << viewed.size << "\n";

        cv::Mat matmul = mask_in * viewed; // (masks_in @ protos.float().view(c, -1))

        // std::cout << "matmul: " << matmul.size << "\n";
        // std::cout << "matmul(0): " << matmul.at<float>(0,0) << "\n";

        cv::MatIterator_<float> it, end;
        for (it = matmul.begin<float>(), end = matmul.end<float>(); it != end; ++it)
        {
            *it = *it / (1 + std::abs(*it)); // fast sigmoid
        }

        std::vector<int> masks_dims{matmul.size[0], mh, mw};
        cv::Mat masks = cv::Mat(masks_dims.size(), masks_dims.data(), matmul.type(), matmul.data); // view(-1, mh, mw)

        // std::cout << "matmul(0) sigmoid: " << matmul.at<float>(0,0) << "\n";
        // std::cout << "masks: " << masks.size << "\n";

        cv::Mat downsampled_bboxes = bboxes.clone();
        // std::cout << "bboxes: "<< bboxes.size << " row:" << bboxes.rows << " bboxes[0]: " << bboxes.row(0) << "\n";
        // std::cout << "bboxes: "<< bboxes.size << " row:" << bboxes.rows << " bboxes[0]: " << bboxes.col(0) << "\n";
        downsampled_bboxes.col(0) = downsampled_bboxes.col(0) * mw / iw;
        downsampled_bboxes.col(2) = downsampled_bboxes.col(2) * mw / iw;
        downsampled_bboxes.col(3) = downsampled_bboxes.col(3) * mw / iw;
        downsampled_bboxes.col(1) = downsampled_bboxes.col(1) * mh / ih;
        // std::cout << "downsampled bboxes[0]: " << downsampled_bboxes.col(0) << "\n";

        // crop_mask
        // masks = crop_mask(masks, downsampled_bboxes) // CHW
        crop_mask(masks, downsampled_bboxes); // CHW
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

        // std::vector<std::pair<ov::Output<ov::Node>, YoloParams>> yoloParams;
        // std::vector<cv::Scalar> colors;
        ColorPlatte colorPlatte = ColorPlatte();

        std::queue<ov::InferRequest> reqQueue = compile(std::move(model),
                                                        FLAGS_m, FLAGS_d, 2 /*roundUp(params.count, FLAGS_bs)*/, core);
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
                std::vector<DetectionObject> objects;

                // MODEL OUTPUT
                const ov::Tensor &box_tensor = req.get_output_tensor(0);
                const ov::Tensor &mask_tensor = req.get_output_tensor(1);
                auto box_shape = box_tensor.get_shape();
                auto mask_shape = mask_tensor.get_shape();

                // INPUT
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

                // RESULTS
                std::vector<int> class_ids;
                std::vector<float> confidences;
                cv::Mat boxes_in;
                std::vector<cv::Rect> boxes_proto;
                cv::Mat masks_in;
                std::vector<cv::Mat> masks_proto;

                // Get output data: out0 -> box, out1 -> mask
                cv::Mat out0 = cv::Mat(NET_LENGHT_OUT0, NUM_BOX, CV_32F, box_tensor.data());
                int out1_shape[3] = {SEG_CHANNELS, SEG_HEIGHT, SEG_WIDTH};
                cv::Mat out1 = cv::Mat(3, out1_shape, CV_32F, mask_tensor.data());

#ifdef DEBUG
                std::cout << "out0: " << out0.size << " out1: " << out1.size << "\n";
                std::cout << "inputhw [" << INPUT_HEIGHT << "," << INPUT_WIDTH << "]\n";
                std::cout << "shape [" << VIDEO_HEIGHT << "," << VIDEO_WIDTH << "]\n";
#endif
                float gain = MIN((float)INPUT_HEIGHT / VIDEO_HEIGHT, (float)INPUT_WIDTH / VIDEO_WIDTH);                          // gain = old / new
                float pad[2] = {(float)(INPUT_WIDTH - VIDEO_WIDTH * gain) / 2, (float)(INPUT_HEIGHT - VIDEO_HEIGHT * gain) / 2}; // wh padding

#ifdef DEBUG
                std::cout << "paddings (" << pad[0] << "," << pad[1] << ")\n";
                std::cout << "gain =" << gain << std::endl;
#endif

                // Filter Box by CONF_THRESH
                for (int i = 0; i < NUM_BOX; i++)
                {
                    // (i: witch box, 4: start at first score (x,y,x,y, box1, box2,...), 1, only take this box, CLASSES: get all class score)
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

                        // // Revert letterbox operation to get real box pos.
                        // int left = (x - pad[0] - 0.5 * w) / gain;
                        // int top = (y - pad[1] - 0.5 * h) / gain;
                        // int width = (w) / gain;
                        // int height = (h) / gain;

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

                std::vector<Detection> _detections{};

                for (unsigned long i = 0; i < nms_result.size(); ++i)
                {
                    int idx = nms_result[i];
                    // segmentation mask
                    masks_in.push_back(masks_proto[idx]);

                    // detection box
                    float rectt[4] = {boxes_proto[idx].x, boxes_proto[idx].y, boxes_proto[idx].width, boxes_proto[idx].height};
                    boxes_in.push_back(cv::Mat(1, 4, CV_32F, rectt));

                    Detection result;
                    result.class_id = class_ids[idx];
                    result.confidence = confidences[idx];

                    result.color = colorPlatte.getColorsByID(result.class_id);
                    result.class_name = CLASS_NAMES[result.class_id];

                    // Revert letterbox operation to get real box pos.
                    auto box = boxes_proto[idx];
                    int left = (box.x - pad[0] - 0.5 * box.width) / gain;
                    int top = (box.y - pad[1] - 0.5 * box.height) / gain;
                    int width = (box.width) / gain;
                    int height = (box.height) / gain;
                    result.box = cv::Rect(left, top, width, height);

                    _detections.push_back(result);
                    DetectionObject obj(result.box.x, result.box.y, result.box.width, result.box.height, result.class_id, result.confidence, result.class_name, result.color);
                    objects.push_back(obj);
                }

                // segmentation mask
                int inputhw[2] = {INPUT_HEIGHT, INPUT_WIDTH};
                process_mask(out1, masks_in, boxes_in, inputhw, true);

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
