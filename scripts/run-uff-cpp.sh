#!/bin/sh
set -e

make
echo

# MODEL_DIR=$HOME/Downloads
# MODEL_FILE=${MODEL_DIR}/hao28-600000-256x384.uff
MODEL_FILE=hao28-600000-256x384.uff

repeat=2
gksize=13

run_batch_example() {
    local BIN=$(pwd)/cmake-build/$(uname -s)/example-batch-detector
    local IMAGES=$(echo $@ | tr ' ' ',')
    local batch_size=4
    ${BIN} \
        --input_height=256 \
        --input_width=384 \
        --batch_size=${batch_size} \
        --use_f16 \
        --gauss_kernel_size=${gksize} \
        --repeat ${repeat} \
        --model_file=${MODEL_FILE} \
        --image_files=${IMAGES}
}

run_stream_example() {
    local BIN=$(pwd)/cmake-build/$(uname -s)/example-stream-detector
    local IMAGES=$(echo $@ | tr ' ' ',')
    local buffer_size=4
    ${BIN} \
        --input_height=256 \
        --input_width=384 \
        --buffer_size=${buffer_size} \
        --use_f16 \
        --gauss_kernel_size=${gksize} \
        --repeat ${repeat} \
        --model_file=${MODEL_FILE} \
        --image_files=${IMAGES}
}

with_images() {
    # local D=$HOME/var/data/openpose
    # $1 \
    #     $D/examples/media/COCO_val2014_000000000192.png \
    #     $D/new-tests/cam0_27.png \
    #     $D/126/cam2_3938.png \
    #     $D/126/cam1_2386.png
    local D=test2017_subset_2/
    $1 \
        $D/000000000016.jpg \
        $D/000000000057.jpg \
        $D/000000000063.jpg \
        $D/000000001730.jpg \
        $D/000000001956.jpg \
        $D/000000002032.jpg \
        $D/000000002063.jpg \
        $D/000000002122.jpg \
        $D/000000002161.jpg \
        $D/000000002253.jpg \
        $D/000000002427.jpg \
        $D/000000002519.jpg \
        $D/000000002799.jpg \
        $D/000000003136.jpg \
        $D/000000003903.jpg \
        $D/000000004001.jpg \
        $D/000000004200.jpg \
        $D/000000004335.jpg \
        $D/000000004398.jpg \
        $D/000000004432.jpg \
        $D/000000004499.jpg \
        $D/000000004790.jpg \
        $D/000000005134.jpg \
        $D/000000005817.jpg
} 

with_images run_batch_example
# with_images run_stream_example
