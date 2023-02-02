![header](https://capsule-render.vercel.app/api?type=rect&color=0:87CEFA,100:4169E1&text=재활용%20품목%20분류를%20위한%20Semantic%20Segmentation&fontSize=32)
<div align="left">
	<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white" />
	<img src="https://img.shields.io/badge/Pytorch-EE4C2C?style=flat&logo=Pytorch&logoColor=white" />
	<img src="https://img.shields.io/badge/OpenMMLab-181717?style=flat&logo=Github&logoColor=white" />
</div>
&nbsp;

# Members
- **김도윤**  : 모델 탐색, UPerNet_BEiT Augmentation 실험 및 Hyperparameter Tuning, Pseudo Labeling 
- **김윤호**  : 모델 탐색, UperNet_ConvNeXt Augmentation 실험 및 Hyperparameter Tuning, Pseudo Labeling 
- **김종해**  : Data EDA, Augmentation 후보 및 조합 탐색, Hyperparameter Tuning, Model Ensemble
- **조재효**  : 모델 탐색, UPerNet_ConvNeXt Augmentation 실험 및 Hyperparameter Tuning, Pseudo Labeling
- **허진녕**  : 모델 탐색, UPerNet_BEiT Augmentation 실험 및 Hyperparameter Tuning, Model Ensemble

&nbsp;

# 프로젝트 개요
> 대량 생산, 대량 소비의 시대에 접어들면서 '쓰레기 대란' 문제가 함께 수면 위로 떠올랐습니다. 많은 쓰레기가 배출되면서 환경 오염 문제가 대두되었고, 이를 해결하기 위해 올바른 분리수거 습관을 함양해야 한다는 목소리가 강해졌습니다. 잘 분리된 쓰레기는 다시 자원으로서의 가치를 인정받기에, 재활용 품목을 분류하는 Semantic Segmentation 모델을 설계하여 환경 부담을 줄이는 과제에 앞장설 것입니다.

&nbsp;

# 데이터셋 구조
```
├─ input
│  ├─ code
│  │  └─ submision
│  └─ data
│     ├─ train
│     │  ├─ images
│     │  └─ mask
│     ├─ valid
│     │  ├─ images
│     │  └─ mask
│     └─ test
│        └─ images
│
└─ Repo
   ├─ CV03
   │  ├─ _base_
   │  ├─ configs
   │  │  ├─ Augmentation
   │  │  ├─ model1
   │  │  └─ model2
   │  └─ utils
   └─ mmsegmentation
```	


&nbsp;

# 프로젝트 수행 절차
<h3> 1. 데이터 EDA  </h3>
<h3> 2. 모델 및 Augmentation 기법 탐색  </h3>
<h3> 3. Baseline 모델 선정 및 최적 Augmentation 기법 선정  </h3>
<h3> 4. Hyperparameter Tuning  </h3>
<h3> 5. Pseudo Labeling </h3>
<h3> 6. Model Ensemble  </h3>

&nbsp;

# 문제정의
<h3> 1. 데이터의 불균형   </h3>  

- Data EDA 결과, 카테고리 별 Annotation 개수에 편차 존재하였다.
- 하지만 카테고리 별 평균 픽셀 수의 경우 거의 균일하였고, 오히려 Annotation 개수가 적은 카테고리 중 하나인 Clothing이 다른 카테고리에 비해 평균 픽셀 수가 약 2배 많았다.
- 카테고리 불균형이 모델 학습에 큰 영향을 주지 않을 것이라 판단하였다.

<h3> 2. 데이터 라벨의 불규칙   </h3>  

- 다른 카테고리에 비해 General Trash는 낮은 IoU, Accuracy를 기록하였는데, 여러 재활용 품목 객체 중 General Trash로 인정될만한 범주가 너무 넓기 때문이라고 판단하였다.
- 객체의 특성을 더 복잡하게 학습해야 한다고 판단하여, 더 복잡한 모델을 가져와 학습에 활용하였고, General Trash의 성능이 향상될 수 있었다.
- 일정 수준 이상으로는 더이상 향상되지 못했는데, 데이터 내 General Trash의 특성이 '무작위성'을 갖기 때문이라고 판단하였다.

&nbsp;

# 모델 및 Data Augmentation
- UPerNet ConvNeXt
	- RandomCutmix (prob=1, patch_scale=(256, 256))  
	 (or CopyPaste (prob=1, mode=all, patch_scale_ratio=0.75))
	- PhotoMetricDistortion
	- RandomFlip (prob=0.5)
	- Normalize
	- RandomRotate (prob=0.8, degree=30)
	- Pad (size=(512, 512), pad_val=0)
   
- UPerNet BEiT Base
	- RandomCutmix (prob=1, patch_scale=(256, 256))  
	 (or CopyPaste (prob=1, mode=all, patch_scale_ratio=0.75))
	- PhotoMetricDistortion
	- RandomFlip (prob=0.5)
	- Normalize
	- RandomRotate (prob=0.8, degree=30)
	- Pad (size=(512, 512), pad_val=0)
	- Resize (img_scale=(640, 640)))

&nbsp;

# Advanced Techniques
<h3> 1. Pseudo Labeling   </h3>  

- 학습에 사용한 이미지 수는 2617장으로, '재활용 품목'을 충분히 대변하지 못하는 양이라고 판단하였다.
- 학습에 사용하지 않은 819장의 test 이미지를 Pseudo Labeling하여 추가 학습에 활용하였고, 성능이 소량 향상하였다.

<h3> 2. Model Ensemble   </h3>  

- Backbone으로서 CNN 계열을 사용하는 UPerNet ConvNeXt와 Transformer 계열을 사용하는 UPerNet BEiT는 상이한 예측결과를 낸다는 점을 활용하여, 두 모델을 토대로 Model Ensemble을 진행하였다.
- 각 픽셀 별로 Hard Voing을 진행하며, 동률인 픽셀에 대해서는 각 픽셀의 카테고리 Accuracy를 비교하여 높은 쪽의 픽셀을 채택하였고, 성능이 소량 향상하였다.