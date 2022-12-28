---
toc: true
badges: true
comments: true

layout: post
keywords: HAR, Skeleton-based, SOTA, PoseConv3D
description: Open Source library mmlab에서 제공하는 기능들 설정
categories: [env, mmlab]
permalink: /setup-mmlab/
title: MMLAB installation guide (non-official)
---

### MMCV-Full with openmim

best practice를 따라 mim으로 mmcv-full을 설치함

```bash
pip3 install -U openmim
mim install mmcv-full
```

### MMPose with pip (for third party)

> [Official Install Guide](https://mmpose.readthedocs.io/en/latest/install.html)

```bash
# pip을 사용해도 무관 (Case b)
pip3 install mmpose
```
#### verify the installation of mmpose 
1. download config and checkpoint
    ```bash
    # 본인이 원하는 폴더 생성
    mkdir verify-mmpose; cd verify-mmpose   # e.g.

    # download
    mim download mmpose --config associative_embedding_hrnet_w32_coco_512x512  --dest .
    ```
2. verify the inference demo
    - pip을 이용해 third party 용 mmpose를 설치했으므로 demo용 python script 생성
        ```bash
        # (optional) option (a)를 따라 /demo 폴더를 생성
        mkdir demo; cd demo
        vi bottom_up_img_demo.py    # open vim
        ```
        ```py
        # bottom_up_img_demo.py
        from mmpose.apis import (init_pose_model, inference_bottom_up_pose_model, vis_pose_result)

        config_file = 'associative_embedding_hrnet_w32_coco_512x512.py'
        checkpoint_file = 'hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
        pose_model = init_pose_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

        image_name = 'demo/persons.jpg'
        # test a single image
        pose_results, _ = inference_bottom_up_pose_model(pose_model, image_name)

        # show the results
        vis_pose_result(pose_model, image_name, pose_results, out_file='demo/vis_persons.jpg')
        ```
3. download image for test
    - 스크립트를 수정하지 않았다면 `verify-mmpose/demo/` 안에 사람이 포함된 `persons.jpg` 를 추가
4. run script
    ```bash
    pwd # /home/devin/env/verify-mmpose
    python3 demo/bottom_up_img_demo.py
    ```
5. 결과

|![demo original image](https://user-images.githubusercontent.com/60145951/209774260-15c23c02-9625-41d8-8e86-34fc6ac6c7f2.jpg)|![demo result image](https://user-images.githubusercontent.com/60145951/209774252-173dbaf3-3b57-4449-a07b-9208d80447bb.jpg)|
|:-:|:-:|
|persons.jpg(input)|vis_persons.jpg(result)|

```bash
└── verify-mmpose
    ├── associative_embedding_hrnet_w32_coco_512x512.py # config
    ├── demo
    │   ├── bottom_up_img_demo.py
    │   ├── persons.jpg
    │   └── vis_persons.jpg
    └── hrnet_w32_coco_512x512-bcb8c247_20200816.pth
```

#### Issue (updated 2022.12.28)

###### [alias expired over ver 1.24] 'numpy' has no attribute 'int'

```bash
# ERROR LOG
Traceback (most recent call last):   File "<stdin>", line 1, in <module>   File "/home/ubuntu/.local/lib/python3.8/site-packages/numpy/__init__.py", line 284, in __getattr__
    raise AttributeError("module {!r} has no attribute " AttributeError: module 'numpy' has no attribute 'int'

# Environment
OS Release              Ubuntu 20.04.4 LTS
mmcv-full               1.7.0
mmpose                  0.29.0
numpy                   >=1.19.5
```

최신버전으로 numpy를 설치했는데 demo를 실행하는 과정에서 `'numpy' has no attribute 'int'` 에러가 발생했다. 공식문서에서 추가하라고 지시한 코드에는 numpy가 없었고, 로그를 살펴보니 이미지의 크기를 정하는 과정에서 int type이 필요했다. 실행한 데모는 bottom_up_transform 이었지만 np.int가 사용된 내역을 살펴보면 bottom_up 뿐 아니라 gesture에서도 사용되는 것 같았다.

```py
# ./datasets/pipelines/gesture_transform.py
# ./datasets/pipelines/bottom_up_transform.py

input_size = np.array([input_size, input_size], dtype=np.int)
```

[stackoverflow](https://stackoverflow.com/questions/74844262/how-to-solve-error-numpy-has-no-attribute-float-in-python) 에 의하면 numpy1.20 부터 np.float 또는 np.int의 alias사용이 중단 되었다. np.int_ 로 대체하거나 int로 변환하라는 권고가 나왔는데, 소스코드를 전부 수정할 수 없는 상황이므로 requirements 에 따라 1.19 로 downgrade 하면 해결된다. 
- NumPy requirements of mmpose: `numpy>=1.19.5`
- [numpy 1.20 relesas note](https://numpy.org/doc/stable/release/1.20.0-notes.html#deprecations)
    > For np.int a direct replacement with np.int_ or int is also good and will not change behavior, but the precision will continue to depend on the computer and operating system. If you want to be more explicit and review the current use, you have the following alternatives:
- [numpy 1.24 relesas note](https://numpy.org/doc/stable/release/1.24.0-notes.html#expired-deprecations) 
    > The deprecation for the aliases np.object, np.bool, np.float, np.complex, np.str, and np.int is expired (introduces NumPy 1.20). Some of these will now give a FutureWarning in addition to raising an error since they will be mapped to the NumPy scalars in the future.

### MMDetection

> [Official Install Guide](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation)

1. 마찬가지로 mim을 통해 mmcv-full을 설치한다.
```bash
pip3 install -U openmim
mim install mmcv-full
```

2. pip으로 mmdet을 설치한다.
```bash
pip3 install mmdet
```

#### Verify installation MMdet

1. download config and checkpoint
    ```bash
    # 본인이 원하는 폴더 생성
    mkdir verify-mmdet; cd verify-mmdet   # e.g.

    # download
    mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
    ```
2. verify the inference demo
    - pip을 이용해 third party 용 mmdet를 설치했으므로 demo용 python script 생성
        ```bash
        # (optional) option (a)를 따라 /demo 폴더를 생성
        mkdir demo; cd demo
        vi img_demo.py    # open vim
        ```
        ```py
        # img_demo.py
        from mmdet.apis import init_detector, inference_detector
        config_file = 'yolov3_mobilenetv2_320_300e_coco.py'
        checkpoint_file = 'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
        model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

        '''
        여기까지가 공식문서의 demo script인데 이렇게 되면 inference_detector된 결과가 CLI환경에 출력되지 않는다.
        https://greeksharifa.github.io/references/2021/08/30/MMDetection/#high-level-apis-for-inference
        위 포스트를 참고하면 결과 이미지를 확인할 수 있다.
        '''
        img = 'demo/demo.jpg'
        inference_detector(model, img)

        result = inference_detector(model, img)
        # visualize the results in a new window
        model.show_result(img, result)
        # or save the visualization results to image files
        model.show_result(img, result, out_file='demo/demo_result.jpg')

        # test a video and show the results
        video = mmcv.VideoReader('demo/demo.mp4')
        for frame in video:
            result = inference_detector(model, frame)
            model.show_result(frame, result, wait_time=1)
        ```
3. prepare for demo
    - mmpose와는 다르게 mmdet에는 demo용 이미지와 동영상이 있으므로 그 파일을 사용하거나 다운로드한다.
    ```bash
    pwd # /home/devin/env/verify-mmdet
    wget https://github.com/open-mmlab/mmdetection/blob/master/demo/demo.jpg ./demo/demo.jpg
    ```
4. run script
    ```bash
    pwd # /home/devin/env/verify-mmdet
    python3 demo/img_demo.py
    ```
5. 결과

|![demo](https://user-images.githubusercontent.com/60145951/209777684-b90f1fe5-58f8-4bae-89a5-de1f796d5215.jpg) | ![demo_result](https://user-images.githubusercontent.com/60145951/209777703-9e673568-dbad-464a-90c7-39ea4a5e7462.jpg) |
| :-: | :-: |
| demo.jpg (input) | demo_result.jpg (result) |


```bash
└── verify-mmdet
    ├── yolov3_mobilenetv2_320_300e_coco.py # config
    ├── demo
    │   ├── img_demo.py
    │   ├── demo.jpg
    │   └── demo_result.jpg
    └── yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth
```
