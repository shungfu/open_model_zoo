// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "graph.hpp"
#include "threading.hpp"

namespace
{
    std::vector<float> paddings(3); // scale, half_h, half_w
    cv::Mat letterbox(cv::Mat &img, std::vector<int> new_shape = {640, 640}, cv::Scalar color = (114, 114, 114), bool autosize = false, bool scale_fill = false, bool scale_up = false, int stride = 32)
    {
        /*
        Resize image and padding for detection. Takes image as input,
        resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints

        Parameters:
            img (np.ndarray): image for preprocessing
            new_shape (Tuple(int, int)): image size after preprocessing in format [height, width]
            color (Tuple(int, int, int)): color for filling padded area
            autosize (bool): use dynamic input size, only padding for stride constrins applied
            scale_fill (bool): scale image to fill new_shape
            scaleup (bool): allow scale image if it is lower then desired input size, can affect model accuracy
            stride (int): input padding stride
        Returns:
            img (np.ndarray): image after preprocessing
            ratio (Tuple(float, float)): hight and width scaling ratio
            padding_size (Tuple(int, int)): height and width padding size
        */

        // Get current image shape [height, width]
        // Refer to https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py#L111

        auto shape = img.size; // current shape [height, width]

        // Scale ratio (new / old)
        float r = std::min(new_shape[1] * 1.0 / shape[1], new_shape[0] * 1.0 / shape[0]);
        if (!scale_up)
        { // only scale down, do not scale up (for better test mAP)
            r = MIN(r, 1.0);
        }

        // Compute padding
        float ratio[] = {r, r};
        int new_unpad[] = {int(round(shape[1] * r)), int(round(shape[0] * r))};
        // wh padding
        float dw = new_shape[1] - new_unpad[0];
        float dh = new_shape[0] - new_unpad[1];
        if (autosize)
        {
            dw = int(dw) % stride;
            dh = int(dh) % stride;
        }
        else if (scale_fill)
        {
            dw = 0;
            dh = 0;
            new_unpad[0] = new_shape[1];
            new_unpad[1] = new_shape[0];
            ratio[0] = new_shape[1] / shape[1];
            ratio[1] = new_shape[0] / shape[0];
        }

        // divide padding into 2 sides
        dw /= 2;
        dh /= 2;

        // Resize and pad image while meeting stride-multiple constraints
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(new_unpad[0], new_unpad[1]));

        // Compute padding boarder
        int top = int(round(dh - 0.1));
        int bottom = int(round(dh + 0.1));
        int left = int(round(dw - 0.1));
        int right = int(round(dw + 0.1));

        // Add border
        cv::copyMakeBorder(resized_img, resized_img, top, bottom, left, right, cv::BORDER_CONSTANT, color);
        // cv::imwrite("/opt/code/demos/multi_channel_object_detection_demo_yolov8/cpp/letterbox.jpg", resized_img);
        return resized_img;
    }
    void framesToTensor(const std::vector<std::shared_ptr<VideoFrame>> &frames, const ov::Tensor &tensor)
    {
        static const ov::Layout layout{"NHWC"};
        static const ov::Shape shape = tensor.get_shape();
        static const size_t batchSize = shape[ov::layout::batch_idx(layout)];
        static const cv::Size inSize{int(shape[ov::layout::width_idx(layout)]), int(shape[ov::layout::height_idx(layout)])};
        static const size_t channels = shape[ov::layout::channels_idx(layout)];
        static const size_t batchOffset = inSize.area() * channels;
        assert(batchSize == frames.size());
        assert(channels == 3);
        uint8_t *data = tensor.data<uint8_t>();
        for (size_t i = 0; i < batchSize; ++i)
        {
            assert(frames[i]->frame.channels() == channels);
            cv::Mat img = letterbox(frames[i]->frame); // resize to (640,640) by letterbox
            // std::cout << frames[i]->frame.size[0] << "/" << frames[i]->frame.size[1] << " --> " << img.size[0] << "/" << img.size[1] << std::endl;
            // cv::resize(frames[i]->frame, cv::Mat{inSize, CV_8UC3, static_cast<void*>(data + batchOffset * i)}, inSize);
            // cv::resize(img, cv::Mat{inSize, CV_8UC3, static_cast<void *>(data + batchOffset * i)}, inSize);
            cv::copyTo(img, cv::Mat{inSize, CV_8UC3, static_cast<void *>(data + batchOffset * i)}, cv::Mat{});
        }
    }
} // namespace

void IEGraph::start(size_t batchSize, GetterFunc getterFunc, PostprocessingFunc postprocessingFunc)
{
    assert(batchSize > 0);
    assert(nullptr != getterFunc);
    assert(nullptr != postprocessingFunc);
    assert(nullptr == getter);
    getter = std::move(getterFunc);
    postprocessing = std::move(postprocessingFunc);
    getterThread = std::thread([&, batchSize]()
                               {
                                   std::vector<std::shared_ptr<VideoFrame>> vframes;
                                   while (!terminate)
                                   {
                                       vframes.clear();
                                       size_t b = 0;
                                       while (b != batchSize)
                                       {
                                           VideoFrame vframe;
                                           if (getter(vframe))
                                           {
                                               vframes.push_back(std::make_shared<VideoFrame>(vframe));
                                               ++b;
                                           }
                                           else
                                           {
                                               terminate = true;
                                               break;
                                           }
                                       }

                                       ov::InferRequest req;
                                       {
                                           std::unique_lock<std::mutex> lock(mtxAvalableRequests);
                                           condVarAvailableRequests.wait(lock, [&]()
                                                                         { return !availableRequests.empty() || terminate; });
                                           if (terminate)
                                           {
                                               break;
                                           }
                                           req = std::move(availableRequests.front());
                                           availableRequests.pop();
                                       }

                                       if (perfTimerInfer.enabled())
                                       {
                                           {
                                               ScopedTimer st(perfTimerPreprocess);
                                               framesToTensor(vframes, req.get_input_tensor());
                                           }
                                           auto startTime = std::chrono::high_resolution_clock::now();
                                           req.start_async();
                                           std::unique_lock<std::mutex> lock(mtxBusyRequests);
                                           busyBatchRequests.push({std::move(vframes), std::move(req), startTime});
                                       }
                                       else
                                       {
                                           framesToTensor(vframes, req.get_input_tensor());
                                           req.start_async();
                                           std::unique_lock<std::mutex> lock(mtxBusyRequests);
                                           busyBatchRequests.push({std::move(vframes), std::move(req),
                                                                   std::chrono::high_resolution_clock::time_point()});
                                       }
                                       condVarBusyRequests.notify_one();
                                   }
                                   condVarBusyRequests.notify_one(); // notify that there will be no new InferRequests
                               });
}

bool IEGraph::isRunning()
{
    std::lock_guard<std::mutex> lock(mtxBusyRequests);
    return !terminate || !busyBatchRequests.empty();
}

std::vector<std::shared_ptr<VideoFrame>> IEGraph::getBatchData(cv::Size frameSize)
{
    std::vector<std::shared_ptr<VideoFrame>> vframes;
    ov::InferRequest req;
    std::chrono::high_resolution_clock::time_point startTime;
    {
        std::unique_lock<std::mutex> lock(mtxBusyRequests);
        condVarBusyRequests.wait(lock, [&]()
                                 {
            // wait until the pipeline is stopped or there are new InferRequests
            return terminate || !busyBatchRequests.empty(); });
        if (busyBatchRequests.empty())
        {
            return {}; // woke up because of termination, so leave if nothing to preces
        }
        vframes = std::move(busyBatchRequests.front().vfPtrVec);
        req = std::move(busyBatchRequests.front().req);
        startTime = std::move(busyBatchRequests.front().startTime);
        busyBatchRequests.pop();
    }

    req.wait();
    auto detections = postprocessing(req, frameSize);
    for (decltype(detections.size()) i = 0; i < detections.size(); i++)
    {
        vframes[i]->detections = std::move(detections[i]);
    }
    if (perfTimerInfer.enabled())
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        perfTimerInfer.addValue(endTime - startTime);
    }

    {
        std::unique_lock<std::mutex> lock(mtxAvalableRequests);
        availableRequests.push(std::move(req));
    }
    condVarAvailableRequests.notify_one();

    return vframes;
}

IEGraph::~IEGraph()
{
    terminate = true;
    {
        std::unique_lock<std::mutex> lock(mtxAvalableRequests);
        while (availableRequests.size() != maxRequests)
        {
            std::unique_lock<std::mutex> lock(mtxBusyRequests);
            if (!busyBatchRequests.empty())
            {
                auto &req = busyBatchRequests.front().req;
                req.cancel();
                availableRequests.push(std::move(req));
                busyBatchRequests.pop();
            }
        }
    }
    condVarAvailableRequests.notify_one();
    if (getterThread.joinable())
    {
        getterThread.join();
    }
}

IEGraph::Stats IEGraph::getStats() const
{
    return Stats{perfTimerPreprocess.getValue(), perfTimerInfer.getValue()};
}
