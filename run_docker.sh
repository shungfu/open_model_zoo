docker run --device /dev/dri:/dev/dri \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -e DISPLAY=$DISPLAY \
            --group-add="$(stat -c "%g" /dev/dri/render*)" \
            -it -u 0 -p 8888:8888 --shm-size 8G -v $(pwd):/opt/code/ \
            --network host --privileged \
            --name model_zoo openvino/ubuntu20_dev
