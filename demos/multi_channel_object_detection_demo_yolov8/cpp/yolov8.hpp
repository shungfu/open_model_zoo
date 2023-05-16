#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

namespace yolov8
{
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

    cv::Mat crop_mask(cv::Mat mask_in, std::vector<float> downsampled_box)
    {
        /*
        It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box
        Args:
            masks_in: [h, w] mask processed mask but without upsample.
            downsampled_box: [4] bbox coordinates in relative point form.
        Returns:
            masked: The mask are being cropped to the bounding box.
        */

        int mh = mask_in.size[0], mw = mask_in.size[1];
        float mx1 = std::min(std::max((float)0, downsampled_box[0]), (float)mw);
        float my1 = std::min(std::max((float)0, downsampled_box[1]), (float)mh);
        float mx2 = std::min(std::max((float)0, downsampled_box[2]), (float)mw);
        float my2 = std::min(std::max((float)0, downsampled_box[3]), (float)mh);

        cv::Mat mask_roi = cv::Mat::zeros(mh, mw, CV_8U);
        mask_roi(cv::Range(my1, my2), cv::Range(mx1, mx2)) = 255;
        cv::Mat masked;
        cv::bitwise_and(mask_in, mask_in, masked, mask_roi);

        return masked;
    }

    float sigmoid_function(float a)
    {
        return 1. / (1. + exp(-1 * a));
    }

    cv::Mat process_mask(cv::Mat protos, cv::Mat mask_in, cv::Mat bbox, int inputhw[2], ov::Shape mask_shape)
    {
        /*
        It takes the output of the mask head, and applies the mask to the bounding boxes. This is faster but produces
        downsampled quality of mask.

        Args:
            protos: [mask_dim, mask_h * mask_w] viewed out of the model.
            masks_in: [1, mask_dim], mask after nms
            bbox: [1, 4], bbox relative to mask after nms.
            intputhw: [INPUT_HEIGHT, INPUT_WIDTH], input shape of the model.
            mask_shape: the size of the mask outputted by the model.
        Returns:
            masked: The processed masks with upsampled shape [inputhw[0], inputhw[1]].
        */

        // int mc = mask_shape[1], mh = mask_shape[2], mw = mask_shape[3];
        int mh = mask_shape[2], mw = mask_shape[3];
        int ih = inputhw[0], iw = inputhw[1];

        cv::Mat matmul = mask_in * protos; // (masks_in @ protos.float().view(c, -1))

        cv::MatIterator_<float> it, end;
        for (it = matmul.begin<float>(), end = matmul.end<float>(); it != end; ++it)
        {
            // *it = *it / (1 + std::abs(*it)); // fast sigmoid
            *it = sigmoid_function(*it);
        }

        cv::Mat results = matmul.reshape(1, 160); // 1x25600 -> 160x160

        std::vector<float> downsampled_bboxes = bbox.clone();

        downsampled_bboxes[0] = downsampled_bboxes[0] * mw / iw;
        downsampled_bboxes[2] = downsampled_bboxes[2] * mw / iw;
        downsampled_bboxes[3] = downsampled_bboxes[3] * mh / ih;
        downsampled_bboxes[1] = downsampled_bboxes[1] * mh / ih;

        cv::Mat masked = crop_mask(results, downsampled_bboxes);

        // if (upsample == true)
        cv::resize(masked, masked, cv::Size(ih, iw), 0., 0., cv::INTER_LINEAR);

        // mask.gt_(0.5)
        for (int r = 0; r < masked.rows; r++)
        {
            for (int c = 0; c < masked.cols; c++)
            {
                float pv = masked.at<float>(r, c);
                if (pv <= 0.5)
                {
                    masked.at<float>(r, c) = 0.0;
                }
            }
        }

        return masked;
    }

    std::vector<cv::Point> mask2segment(cv::Mat mask_float)
    {
        /*
        It takes a single mask(h,w) and returns a segment(xy)

        Args:
            mask_float: processed mask, which is upsampled shape (640, 640)

        Returns:
            segments: segment mask
        */

        cv::Mat mask_int;
        mask_float.convertTo(mask_int, CV_8U);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask_int, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        std::vector<cv::Point> max_contour;

        if (!contours.empty())
        {
            int max_len = 0;
            int max_idx = 0;
            // Find the index of the contour with max len
            for (int i = 0; i < contours.size(); i++)
            {
                if (contours[i].size() > max_len)
                {
                    max_len = contours[i].size();
                    max_idx = i;
                }
            }
            max_contour = contours[max_idx];
        }

        return max_contour;
    }

    std::vector<cv::Point> scale_segment(float gain, float pad[2], std::vector<cv::Point> segment, int VIDEO_HEIGHT, int VIDEO_WIDTH)
    {
        /*
        Rescale segment coordinates (xyxy) from input shape (INPUT_HEIGHT, INPUT_WIDTH) to video shape (VIDEO_HEIGHT, VIDEO_WIDTH),
        and clips them to the video shape.

        Args:
            gain: scaled factor same as the boundingbox.
            pad: pad size same as the boudingbox
            segment: the segment to be scaled and cliped.
            VIDEO_HEIGHT: target height.
            VIDEO_WIDTH: target width.
        Returns:
            segments (vector<Point>): the segmented image.
        */

        for (int i = 0; i < segment.size(); i++)
        {
            segment[i].x = std::min((int)((segment[i].x - pad[0]) / gain), VIDEO_WIDTH);
            segment[i].y = std::min((int)((segment[i].y - pad[1]) / gain), VIDEO_HEIGHT);

            // segment[i].x = (segment[i].x - pad[0]) / gain;
            // segment[i].y = (segment[i].y - pad[1]) / gain;
        }
        return segment;
    }

}