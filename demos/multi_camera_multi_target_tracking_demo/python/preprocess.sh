# download inference video
if [ ! -f people.mp4 ]
then
    wget https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/data/video/people.mp4
fi

# download model
if [ ! -d intel ]
then
    omz_downloader --name person-detection-retail-0013
fi

# copy yolov8n-segmentation model
if [ ! -d yolov8n-seg_openvino_int8_model ]
then
    cp -r $HOME/yolov8/yolov8n-seg_openvino_int8_model/ .
fi