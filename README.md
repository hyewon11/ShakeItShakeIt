StyleGan2-ADA을 이용한 커스터마이징 2세 예측 프로그램:   
Shake It, Shake It!
=======================================================================

## Play It!
당신의 2세 얼굴을 만들어보세요!    
'Shake It, Shake It!' Colab: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1Um6E6OOvlxqoK7pNjxMdq2nakldLu4No/view?usp=sharing)    

## 'Shake It, Shake It!'이란
**Shake It, Shake It은 StyleGan2-ADA를 기반으로 학습한 2세 예측 프로그램입니다.**    
총 20000장의 데이터에 대하여 동양인과 ffhq데이터 셋의 비율을 다양하게 조절하여 StyleGan2-ADA 모델로 generator network를 학습시켰습니다. 그 중 가장 성능이 좋은 network 피클을 이용하여 encoder부분에서 남녀 이미지 의 latent vector를 추출하였고, 그 두개의 벡터를 합성시킨 후 성별, 연령 등의 스타일 direction vector를 추가로 합성시켜 2세 예측 이미지를 생성하였습니다.   

Based on: [StyleGan2-ADA](https://github.com/NVlabs/stylegan2-ada)   
Surgery: [StyleGAN2-surgery](https://github.com/aydao/stylegan2-surgery)   
Encoder: [StyleGAN2-ada-Encoder](https://github.com/hyewon11/stylegan2-ada-encoder)([StyleGAN-encoder](https://github.com/Puzer/stylegan-encoder) 참고)      

**어떻게 프로젝트를 진행했는지 더 알고 싶다면?, Click it!:** [Notion](https://www.notion.so/Finishing-the-Project-Shake-it-Shake-it-a9620e1b95794bdc96ad115a0cca95ef)

## 결과
<p align="center" style="color:gray">
  <img style="margin:50px 0 10px 0" src="https://user-images.githubusercontent.com/39722108/104934088-120e9980-59ed-11eb-8a03-a7ca42b83020.PNG" alt="ER diagram" width="80%" height="80%"/>
</p>

**더 많은 결과 이미지를 보고 싶다면? Click it!:** [Google Drive]()

## 의의
* 256*256 data와 StyleGan2-ADA 모델 사용으로 효율성 증대
* 자녀의 성별 선택, 남녀 비율 조절, 화장 제거 기능
* 동양인에 최적화 되면서도 서양인에게도 적절하게 적용되는 모델 구현

## 한계점
* 남자, 어린아이 데이터가 여자데이터에 비해 제한적
* 3-5세의 매우 낮은 연령대는 잘 생성되지 않음
* 낮은 배경 반영률 + 이전보다 사진과 머리,피부 톤이 더 어두워 지기도 함
* 데이터셋에 따른 결과물의 완성도 차이 발생

## Dataset
* FFHQ Dataset
* Asian Face Dataset

## 개발 환경 및 개발 언어
* 개발 환경   
**Colab Pro**: GPU(T4 or P100 or V100), CPU(Intel Xeon CPU @ 2.30GHZ), RAM(25.51GB)   
* 개발 언어   
![issue badge](https://img.shields.io/badge/tensorflow-1.15-yellow)

## Contributor
* 강민지
* 진정민
* 진혜원
* 한하랑
