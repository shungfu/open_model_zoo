python3 multi_camera_multi_target_tracking_demo.py \
    -m intel/person-detection-retail-0013/FP16-INT8/person-detection-retail-0013.xml \
    --m_yolov8seg yolov8n-seg_openvino_int8_model/yolov8n-seg.xml \
    -d GPU \
    --loop \
    -i people.mp4
    # -i /opt/code/demos/multi_channel_object_detection_demo_yolov8/cpp/coco_bike.jpg
    