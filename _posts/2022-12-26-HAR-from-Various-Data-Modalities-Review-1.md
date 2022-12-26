---
toc: true
badges: true
comments: true

layout: post
keywords: HAR, Review
description: 
categories: [HAR, review/survey]
permalink: /HAR-from-Various-Data-Modalities-Review-1/
title: Human Action Recognition from Various Data Modalities; A review (1)
---

#### focused and reviewed
1. mainstream deep learning architectures
    - single modalities
    - multiple modalities for enhanced HAR
2. DATA : short trimmed video segments
    - one, only action instance
3. Benchmark Datasets

#### Contribution follows
1. various data modalities
2. multi-modality based HAR
    - approach 1: fusion based
    - approach 2: cross modality co-learning-based
3. recent, advanced methods: SOTA approaches
4. comprehensive comparison of existing method

## 1. introduction
> 다양한 data modalities들의 장점, 한계를 및 modality간의 연구 흐름 파악


기술의 발전과 방법론의 창안은 선행 연구의 한계와 발전 가능성에 기초하므로 기술 발전의 흐름과 맥락을 숙지하는 과정은 중요하다. 본 리뷰 논문은 2022 IEEE에 발표된 Review 논문으로 다양한  인간 행동 표현형을 인식하는 HAR 연구의 최신 흐름을 기술하고 있다. 개요에 따르면 인간 행동은 다양한 데이터 양식으로 표현될 수 있다.

| visual modalities | non-visual modalities |
| :---------------: | :-------------------: |
| RGB               | audio    
| Skeleton          | acceleration  
| depth             | radar     
| infrared          | wifi signal
| point cloud $^{1}$|             
| event stream      |             

* point cloud $^{1}$ 3차원 공간상에 퍼져 있는 여러 포인트(Point)의 집합(set cloud)으로 Lidar 센서와 RGB-D 센서로 수집된 데이터다.

위 데이터 양식은 크게 가시성 `visibility` 에 따라 `visual modalities`, `non-visual modalities` 로 나뉜다. 


### 1.1 visual modalities  

일반적으로는, visual modalities가 HAR 발전에 큰 영향을 미쳐왔다. HAR의 많은 발전이 RGB Video 또는 images를 기반으로 이루어졌음을 보면 알 수 있다. `RGB data`는 관측 (surveillance) 또는 추적 (monitoring) 시스템에서 보편적으로 사용되어왔다. RGB 는 기본적으로 세 채널을 가진 '이미지(들)'이기 때문에 큰 computing resource 를 필요로 하는데 이를 보완하기 위해 사용된 데이터가 `skeleton` 이다. skeleton data는 인간 관절의 움직임 (trajectory of human body joints) 을 encoding한 데이터로 간명하고 효과적이다. 그러나 물체가 포함되어있거나 장면간 맥락을 고려해야 하는 경우 skeleton data 만으로는 정보를 충분히 얻을 수 없는데 이 때 `point cloud` 와 `depth data`를 사용한다. 또한, '본다'는 행위는 근본적으로 빛에 의존하는데 이 한계를 극복하기 위해 `infrared data`를 사용하며 `event stream` 은 불필요하게 중복(redundancy)되는 데이터를 

1. 가시영역에서 얻을 수 있는 정보를 얻거나 : RGB
2. 인간이 눈으로 보고 이해하는 정보에 **집중**하여 인간 신체에서 이른바 ROI를 추출해내는 방법을 적용한다: skeleton
3. 사람과 환경의 상호작용 또는 시공간적 맥락을 고려하기 위해 3D 구조에서 주요한 정보를 추출하는 작업을 거친다: point cloud, depth data
4. 시각에 의존하지 않고 나아가 비가시영역인 적외선 영역에서 정보를 얻음으로써 **빛에 의존해야 한다는** visual modalities의 태생적 한계를 극복한다: infrared data
5. 불필요한 중복, 또는 정보를 제거하여 HAR에 적합한 데이터를 구축한다: event stream

> 정보를 얻고, insight 또는 활용 가능한 부분만 추려내어 새로운 방법론을 창안하고, 기술이 최적화 되었을 때 더 필요한 정보를 얻기 위해 또 다른 modality를 사용하며 다시 불필요한 부분을 제거하는 방식으로 기술이 발전되어왔다.

### 1.2 non-visual modalities  

눈으로 봤을 때 직관적이지 않지만 사람들의 행동을 표현하는 또 다른 방식이다. 직관적이지 않음에도 사용될 수 있는 이유는 특정 상황에서 대상의 **개인정보 보호**가 필요할 때다. `audio` 는 시간에 따른 상황 (temporal sequence) 에서 움직임을 인지하기에 적절하며, `acceleration` 는 fine-grained HAR에 사용된다. $^2$ 또한 `radar`는 물체 뒤의 움직임도 포착할 수 있다.

$^2$ fine-grained HAR : 세분화된 HAR. acceleration data가 이에 사용된다는 내용은 영상의 움직임에서 가속도를 알아내는 방향의 연구를 말하는 것으로 보인다. ([관련 논문](https://arxiv.org/abs/2211.01342))

### 1.3 Usage of modalities (single vs multi)

#### single modality and effect of their fusion

살펴본 바와 같이, 각 modality는 서로 다른 강점과 한계를 가지고 있으므로 여러 양식의 데이터들을 합치거나(`fusion`), 데이터 양식간 'transfer of knowledge'$^3$ 를 진행하여 정확도와 강건함을 높인다.  
나아가 fusion은, 서로 다른 두 개 이상의 데이터에서 각 데이터간 장단점을 상호보완하기 위한 방법론으로 소리 데이터와 시각 데이터를 포함함으로써 단순히 '물건을 내려 놓는' label을 가방을 내려 놓는지, 접시를 내려놓는지 구체적으로 구분할 수 있게 한다. 

$^3$ `Transfer learning` 은 Transfer Learning과 Knowledge Distillation으로 나뉘는데 ([ref](https://baeseongsu.github.io/posts/knowledge-distillation/#etc-%EA%B7%B8-%EB%B0%96%EC%97%90)) 서로 다른 도메인에서 지식을 전달하는 방법이 Transfer Learning (fine tuning 필요) 이고, 같은 도메인에서 다른 모델 간 지식 전달이 이루어지는 것을 Knowledge Distillation이라고 하면 'transfer of knowledge across modalities'는 Transfer Learning을 말하는 것으로 보인다.

#### data modalities and its pros and cons

<div align="center">
<img width="661" alt="image" src="https://user-images.githubusercontent.com/60145951/209530662-e526a0ee-58a4-4ad1-abf2-34ad0bb32c5c.png">
</div>

