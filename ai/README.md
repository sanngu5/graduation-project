# 동영상 속 캡션 제거 및 복원

## 팀명단
- 이종우
- 박성진
- 윤진성

## 1학기 결과물

### 요구사항 분석서
https://docs.google.com/document/d/1ctgq-h6bQqH5MkAv5bFROycGwb25swo49zGy9lU6q5c/edit

### 시스템 설계서
https://docs.google.com/document/d/14tE19qj9MCoHoC1SMsLbqWDbHi_3VDtoxljbQDJ3xuM/edit

### 프로토타입
https://youtu.be/JF13KG4Gyi8

### 기말 보고서
https://docs.google.com/document/d/1qcoYHLGHmuc_UCf8TaPu4rp1sqGg9izVu9atNjeriUI/edit

## AI 실행환경
1. conda 환경 설정
```
conda env create -f environment.yml 
conda activate cpp
```

2. ffmpeg 설치
* ffmpeg (video to png)


## IQA(cal3.py) 실행환경
```
conda create -n iqa python=3.8 -y
conda activate iqa

conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch

pip install pyiqa
pip install pillow
pip install tqdm
pip install numpy
```

## Citation
If any part of our paper and repository is helpful to your work, please generously cite with:
```
@inproceedings{chu2021deep,
  title={Deep Video Decaptioning},
  author={Chu Pengpeng, Quan Weize, Wang Tong, Wang Pan, Ren Peiran and Yan Dong-Ming},
  booktitle = {The Proceedings of the British Machine Vision Conference (BMVC)},
  year={2021}
}
```