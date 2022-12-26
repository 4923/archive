---
toc: true
badges: true
comments: true

layout: post
keywords: HAR, Review, single modality
description: 데이터의 한계와 장점을 최소/최대화 하기 위한 single modalities methods
categories: [HAR, review/survey, single modality]
permalink: /HAR-from-Various-Data-Modalities-Review-2/
title: Human Action Recognition from Various Data Modalities; A review (2)
---

## 2. Single modality
RGB, Skeleton, depth, infrared, point cloud, event stream, audio, acceleration, radar, WiFi will be reviewed

## visible Modalities

### 2.1 RGB

> 한계와 장점 모두 RGB Data의 특성에서 비롯된다.

- 특성
    1. 이미지(들) 로 이루어져있다. ($\because$ video is sequence of images)
    2. RGB data를 생성하는 카메라는 사람의 눈으로 보는 장면을 재생산하는 것을 목적으로 한다.
    3. 수집하기 쉽고 상황과 맥락을 반영하고 있는 풍부한 외관 정보가 포함되어있다.
        - 'rich appearance information'
    4. 폭넓은 분야에 사용될 수 있다.
        - visual surveillance: 사람이 한 순간도 놓치지 않고 관찰할 수는 없는데 이를 보완할 수 있다.
        - autonomous navigation: 자율주행(ANS)의 일부로써 사람의 개입 없이 정확하게 목적지까지 도달하도록 하는 기술이다.
        - sport analysis: 눈으로 쫓기 힘든 순간들을 정밀하게 판독해야 하는 분야이므로 이 또한 '사람의 눈'을 대신한다.
- 한계
    - various of background, viewpoints, scales of humans
        - 학습할 수 있는 데이터는 한정적이고, 이를 활용할 수 있는 변수는 너무 많다.
    - illumination condition
        - 촬영이라는 개념이 갖는 근본적인 한계로, 광원 상태에 따라 결과가 달라질 가능성이 있다.
    - high computational cost
        - 영상은 이미지의 연속이므로 공간과 시간을 동시에 고려하여 모델링하려면 많은 자원이 요구된다.

- modeling methods
    1. pre-deep learning : handcrafted feature-based approach, 수작업 특징 기반 접근법
        - Space-Time Volume-based methods
        - Space-Time Interest Point (STIP)
    2. deep learning : currently mainstream
        - backbone model을 무엇으로 사용하느냐에 따라 나뉠 수 있다. 
        1. two-stream CNN based method / multi stream architectures (extension of two stream)
            - backbone : 2D CNN
            - 시간정보가 포함될 수 밖에 없기 때문에 temporal information, spatial information 모두 고려하는 two-stream 접근이 제안되었다.
        2. RNN based method
            - feature extractor : RNN model with 2D CNNs
            - RGB-based model
        3. 3D CNN based method

#### 2.1.1 Two Stream 2D CNN-Based Methods
> two 2D CNN branches taking different input features extracted from RGB video for HAR and the **final result** is usually obtained through **fusion strategy**

**classical approach**

고전적으로 two stream network는 각 network를 병렬적으로 학습시킨 후 결과를 융합 `fusion` 하여 최종 결과를 추론했다. 예를들어, input이 video이면 video에 내재된 정보들을 크게 1) rgb 프레임들은 공간 네트워크의, 2) multi-frame-based optical flow는 시간 네트워크의 학습 정보가 된다. 각 stream은 1) 모양 특성, appearance feature 과 2) 동작 특성 motion feature을 각각 학습한다.

- multi-frame-based optical flow: 움직임을 묘사하는 방법으로, 주로 짧은 시간 간격을 두고 연속된 이미지들로 구성된다. optical flow는 이미지의 velocity(속도) 를 계산하는데 이 속도는 이미지의 특정 지점이 다음의 어디로 이동할지 예측할 수 있게 한다. 
    - 주로 video understanding 에서 사용되는 개념으로 보인다.
    - acceleration data와 어떻게 다른지 알 필요가 있다.
    > [The optical flow is used to perform a prediction of the frame to code](https://arxiv.org/pdf/2008.02580.pdf)
    - networks: SpyFlow, PWC-Net; compute pixel-wise motion vectors

**overcome limitation**

RGB 양식 데이터를 사용함에 있어 주된 문제로 지적되는 점은 '큰 데이터 용량으로 인한 computing resource 부담과 연산 속도 저하'이므로 연산 속도를 높이기 위해 해상도를 낮추거나, 고해상도 데이터에서 center crop을 하는 기법을 적용했다.

**better data representation**

모델의 성능은 데이터의 양과 **질**에 좌우된다. 따라서 더 나은 video representation 에 눈을 돌리게 된다. Wang은 multi-scale video frames, optical flow를 각 CNN stream에 넣어 특성맵 feature map을 추출했고 이에서 trajectories에 중심을 둔 **spatio-temporal tubes** (`action tube`)를 샘플링했다. 이렇게 한 결과, Fisher Vector representation $^{3)}$ 과 SVM을 통해 분류했다.

- 왜 튜브일까? : $2)$ 가 시기상 더 먼저 나온 논문이므로 후자에서 제시된 개념일 것으로 추정됨. 후자 논문의 Abstract에서 tube는 "예측된 동작을 연결함으로써 시간일관적으로 객체를 탐지하는" 개념이다. 
    1. suggest image region : 움직임이 두드러지는 영역을 선택
    2. CNN을 이용하여 공간적 특징을 추출
    3. Action tube를 생성

    > We link our predictions to produce detections consistent in time, which we call action tubes.    

    <figure>
    <img width="834" alt="image" src="https://user-images.githubusercontent.com/60145951/209574143-74cb413a-7e2f-449f-ba72-6624404b8904.png"/>
    <figcaption>Fig 1. Discovering Spatio-Temporal Action Tubes; An over view of action detection framework </figcaption>
    </figure>

    <figure>
    <img width="815" alt="image" src="https://user-images.githubusercontent.com/60145951/209574675-7829d4c4-f9d6-4e5a-89d0-59596ca8ff9e.png"/>
    <figcaption>Fig 2. Finding action tubes; action tube approach, detect action on (a) and link detected actions in time to produce action tube*s*</figcaption>
    </figure>

- $^{1)}$ [Discovering Spatio-Temporal Action Tubes (2018)](https://arxiv.org/abs/1811.12248)
- $^{2)}$ [Finding action tubes](https://ieeexplore.ieee.org/document/7298676) or [CVPR open access](https://openaccess.thecvf.com/content_cvpr_2015/papers/Gkioxari_Finding_Action_Tubes_2015_CVPR_paper.pdf)
- $^{3)}$ Fisher Vector Representation: [ref](https://zlthinker.github.io/Fisher-Vector)

**Long term video level information**

> 정보를 mean pooling하거나 누적하여 단일한 움직임이 아닌 움직임의 연속; 좀 더 복잡한 행동을 인식

각 비디오를 세 개의 segment로 나눈 후 two stream network에 입력한 후, 각 segment의점수를 **average pooling** 을 이용해 융합한다. 또는 segment 점수를 pooling하지 않고 **element-wise multiplication**으로 특성의 총계를 구한다. 이 때 two stream framework에 의해 샘플링된 외형과 동작 프레임들은 '하나의 video-level multiplied'를 위해 aggregate 연산되며 이를 `action words` 라고 칭한다. $^{4)}$ 

<figure>
<img width="618" alt="image" src="https://user-images.githubusercontent.com/60145951/209576243-dfe0955c-7e9b-4375-88e1-f236719d6273.png"/>
<figcaption>Fig 3. 동작들에서 행동과 관계된 `action word`를 추출한 후 이를 총 망라하는 하나의 분류를 선택하는 과정</figcaption>
</figure>

- $^{4)}$ R. Girdhar, D. Ramanan, A. Gupta, J. Sivic, and B. Russell, “Actionvlad: Learning spatio-temporal
aggregation for action classification,” in CVPR, 2017.

**EXTENSION of two stream CNN based method** 

3 stream 으로 확장하는 등, "움직임" 또는 "프레임간 연속성"을 학습시키기 위해 다양한 방법론을 도입했다. 이후 2 stream siamese network (SNN) 로 확장되었는데 이는 동작 발생 **전**과 동작 **후** 프레임에서 특징을 추출하는 `one shot learning`의 일종으로 연속성이 아닌 동작 시작, 전, 후를 구분하여 학습하는 발상의 전환을 꾀한다.

- one shot learning : 소량의 데이터로 학습할 수 있게 하는 방법이 few shot learning이라면 one shot은 그 극한으로 이미지 한장을 학습 데이터로 삼는 방법론이다. 사람은 물체간의 유사성을 학습하는데, 이 유사성은 물체를 배우고 물체간의 유사성을 또 다시 배우는 과정으로 나뉠 수 있다. 다시말해, **물체의 특성을 학습하고 이를 일반화**할 줄 아는 능력을 학습시키는 방법이 one/few shot learning이다. 
    - $\therefore$ 이미지 자체의 특성을 학습하는 것이 아닌, 이미지간의 유사성을 파악하고 유사도를 파악할 때 쓰는 기법인 '거리 함수'를 사용한다.
    
- 샴네트워크 
    <figure>
    <img width="600" alt="siamese network" src="https://user-images.githubusercontent.com/60145951/209576942-b98a7973-be82-46f3-b292-cbf6be463153.png"/>
    <figcaption>Fig 4. <a href="https://serokell.io/blog/nn-and-one-shot-learning">ref: A Guide to One-Shot Learning</a></figcaption>
    </figure>
    - two stages: verification and generalization 가 포함된다.
    - 각각 다른 입력을 동일한 네트워크 인스턴스에 학습시키고, 이는 동일한 데이터셋에서 훈련되어 유사도를 반환한다. 

**Tackle high computational cost**

Knowledge distillation $^{5)}$ 이 사용된다. "[Data 에서 의미를 파악하면 Information 이 되고, Information 에서 문맥을 파악하면 Knowledge 이 되고, Knowledge 를 활용하면 Wisdom 이 된다.](https://velog.io/@dldydldy75/%EC%A7%80%EC%8B%9D-%EC%A6%9D%EB%A5%98-Knowledge-Distillation)" 모델 압축을 위한 절차로, soft label과 hard label을 일치시키는 것이 목적이며 soft label에는 temperature scaling function을 적용하여 확률 분포를 부드럽게 만든다. 예를 들어 feature들의 label이  $[0, 1, 0]^{T}$ 이면 Hard label, $[0.05, 0.75, 0.2]^{T}$ 이면 soft label이다. 각 feature들은 서로 다른 특성을 가지고 있지만 공통된 특성 또한 가지고 있기 때문에, 이 공통 요소를 포함하는 class score를 날려버리면 (hard label) 정보가 손실되는 셈이다. 이렇게 정보가 손실되지 않게 Teacher network를 구성하고 Student network가 teacher network에 최대한 가까운 정답을 반환하도록 학습시킨다. 위에서 언급한 `temperature`는 그 값이 낮을 때 입력값의 출력을 크게 만들어주는 등 필요에 따라 값에 가중치를 둠으로써 Soft label의 이점을 최대화 한다. [참고](https://light-tree.tistory.com/196)

| teacher network; optical flow data | 
| :--------------------------------: |
|     ⬇︎ Knowledge Distillation ⬇︎     |
|   student network; motion vector   |

| ![](https://velog.velcdn.com/images%2Fdldydldy75%2Fpost%2F2bc5e3eb-b58b-456b-9b6b-420cd996ae38%2Fimage.png) | ![https://intellabs.github.io/distiller/knowledge_distillation.html](https://intellabs.github.io/distiller/imgs/knowledge_distillation.png) |
|:-:|:-:|
| Knowledge Spectrum | Distillation Architecture |


- $^{5)}$ [Distilling the Knowledge in a Neural Network(2015)](https://arxiv.org/abs/1503.02531)

**In Conclusion,**

여러개의 stream으로 CNN architecture들을 확장하거나 더 깊게 레이어를 쌓는 등 여러 시도를 해보았으나 수많은 video의 frame 개수를 고려할 때 깊이는 오히려 HAR에 방해가 될 수 있다. 선행 연구를 통해 '차별화된 특징'을 예측하는 것이 중요함을 파악하게 되었다. 이 외에도, fusion strategy research의 마지막 conv layer에서 공간과 시간 네트워크를 융합하는 방법이 위에서 지적된 컴퓨팅 자원을 절약하면서 (params를 줄이면서) 정확도를 유지하는 효과적인 방법임을 알아냈다.

#### 2.1.2 RNN based

> feature extractor 로 CNN을 사용한 hybrid architecture

**LSTM based model**

Vanilla Recurrent Neural Network의 gradient vanishing 문제로 인해 RNN based solution은 gate 를 포함하는 RNN Architecture를 채택한다. (e.g. LSTM)

<figure>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/60145951/209581127-eba182df-90a6-4c9c-b146-268ebb92f144.png"/>
<figcaption>Fig 5. RGB modality modeling methods (CNN, RNN based)</figcaption>
</figure>

물론 '이미지'에서 공간적 요소를 빠트릴 수 없기 때문에 특징 추출은 여전히 **2D CNN**으로 진행하고, 시간요소를 LSTM에서 차용한 구조를 통해 모델링한다. 이를 `LRCN` (Long Term Recurrent Convolutional Network, Jeff Donahue et al. in 2016) 라고 하며 이는 '2D CNN; 프레임 단위 RGB feature 추출' + 'label 생성 LSTM'으로 구성된다. 


| ![LRCN architecture](https://kobiso.github.io//assets/images/lrcn.png) | ![recognition examples](https://kobiso.github.io//assets/images/lrcn_tasks.png) |
| :-: | :-: |
| Fig 6.1 LRCN architecture | Fig 6.2 LRCN model for some tasks |

- image reference : [Long-term Recurrent Convolution Network(LRCN)](https://kobiso.github.io/research/research-lrcn/)

**attention mechanism**

multi layer LSTM model 설계 후 다음 프레임에 가중치를 부여한 Attention map 을 **재귀적으로** 출력함으로써 공간적 특성에 집중할 수 있게 되었다.
- feature map을 중첩해서 뽑아내는 것 처럼?
- recap : main idea of attention; decoder에서 출력 단어를 추론하는 매 순간마다 encoder에서의 전체 문장을 참고한다는 점. 단, '해당 시점에서 예측에 필요한 부분에 집중(attention)'해서 구한다. $Attention (Q, K, V) = Attention \ Value$ 로, Query에 대해 모든 Key와의 유사도를 구한 후 (*전체 문장을 참고*) 이에 관계된 Value에 반영한다. 유사도가 반영된 값, value는 attention value라고도 한다.
    - Q : t 시점에서 디코더 셀에서의 은닉상태
    - K : keys, '모든 시점에서' 인코더 셀의 은닉 상태
    - V : Values, '모든 시점에서' 인코더 셀의 은닉 상태로 각 인코더의 attention 가중치와 은닉상태가 가중합 된 값이다. (a.k.a. context vector)

#### 2.1.3 3D CNN based method

> HAR의 공간과 시간을 모두 식별할 수 있다는 강력한 장점이 있으나 많은 양의 훈련 데이터를 요구함

> 한계: Long term Temporal information이 전해지지 않는 문제

지금까지는 모두 2D CNN을 시간과 함께 모델링했다. 그러나 Tran et.al [66]은 raw video data에서 시공간 데이터를 end-to-end 학습하기 위해 3D CNN 모델을 도입한다. 단, 이 경우 클립 수준 (16 frames or so) 에서 사용되는 모델이므로 시간이 길어질수록 temporal 정보가 옅어지는 한계가 있다.  
이에, Diba et al.([github](https://github.com/MohsenFayyaz89/T3D), [paper](https://paperswithcode.com/paper/temporal-3d-convnets-new-architecture-and))는 3D 필터 및 pooling kernel로 2D 구조였던 DenseNet을 확장한 T3D (Temporal 3D ConvNet) 과 새로운 시간 계층 TTL (Temporal Transition Layer) 을 제안했다. 
- 시간에 따라 convolution kernel depth가 달라지도록 모델링 한 것 같다. 3D CNN 모델이 2D 단위에서 학습한 것을 활용하지 않는 것에 착안해 2D와 3D를 함께 쓰는 방식을 채택한 것으로 보임 (논문까지 확인하기 전)

그 외에, 시간 범위를 늘린 LTC (Long-term Temporal Convolution) 모델, multi scale 'temporal-only' convolution; Timeception 모델 등이 제안되었으며 이는 모두 복잡하거나 긴 작업에서 영상의 길이에 구애받지 않고 인식할 수 있는 강건한 모델을 만들기 위함이다.



### 2.2 Skeleton

시점 변화에 민감한 pose estimation에서 motion capture system으로 수집한 데이터셋은 신뢰할 수 있다. (“Ntu rgb+d 120: A large-scale benchmark for 3d human activity understanding,” TPAMI, 2020.) 최근의 많은 연구는 Ntu의 depth map, 또는 RGB video를 사용한다 (st-gcn).

RGB video만 사용할 경우 옷 또는 신체의 부피를 포함해 RGB data의 문제였던 다양한 변수 (e.g. background, illumination environment) 로부터 상당부 자유로울 수 있다. 초기에는 수작업으로 특징을 추출하여 관절 또는 신체 부위 기반의 방법이 제안되었는데 딥러닝의 발전에 따라 RNn, CNN, GNN, GCN을 적용하게 되었다.

<figure>
<img width="539" alt="image" src="https://user-images.githubusercontent.com/60145951/209585313-6fc0b53c-f815-4b60-baa0-13c677a06ab5.png">
<figcaption>Fig 7. Performance of skeleton-based deep learning HAR methods on NTU RGB+D and NTU RGB+D 120 datasets.</figcaption>
</figure>

### 2.3 depth

> Depth maps refer to images where the pixel values represent the distance information from a given viewpoint to the points in the scene. 

색상, 질감 등의 변화에 강건하며 3차원 상의 정보이므로 신뢰할 수 있는 3D 구조 및 기하학적 정보를 제공한다. depth map은 왜 필요한가? 3D 데이터를 2D 이미지로 변환하기 위함이다: depth image  
depth 정보는 특수한 센서를 필요로 하는데, 이는 active sensors (e.g., Time-of-Flight and structured-light-based cameras) and pas- sive sensors (e.g., stereo cameras) 로 나뉜다.   
active sensor는 방사선을 물체에 방출하여 **반사되는 에너지를 측정**하여 깊이정보를 얻는, 말그대로 능동적인 행동에 의해 발생하는 정보를 수집하는 센서다. Kinect, RealSense3D 등의 특수한 장치를 포함하는 센서가 포함된다.
passive sensor는 물체가 방출하거나 반사하는 **자연적인 에너지** 를 말한다. 수동센서의 예인 stereo camera는 인간의 양안을 시뮬레이션 하는 카메라로 are recovered by seek- ing image point correspondences between stereo pairs 한다. 

둘을 비교했을 때, passive depth map generation은 RGB 이미지 사이에서 깊이를 연산해내는 과정이 포함되므로 계산 비용이 많이 들 뿐 아니라 질감이 없거나 반복 패턴이 있는; view point에 따라 크게 달라지지 않는 대상에는 효과를 보이지 않을 수 있다. 따라서 대부분의 연구는 active sensor를 이용한 depth map에 초점을 맞추고 있다.

### 2.4 infrared

### 2.5 Point Cloud

| ![lidar point cloud](https://miro.medium.com/max/1400/1*Gbzp4-b8zXe5JGZmG-uXNw.webp) |
| :-: |
| Fig 1. lidar point cloud |

### 2.6 event stream

## Non-visible Modalities

### 2.7 audio

### 2.8 acceleration

### 2.9 radar

### 2.10 WiFi