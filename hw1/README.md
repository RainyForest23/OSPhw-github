# Readme 파일
OSP 과제 1 - 이미지 회전 및 스티칭

## 개요
이 과제에서 구현한 두 가지 이미지 처리 기법:
1) 이미지 회전 - Nearest Neighbor와 Bilinear 보간법 사용
2) 이미지 스티칭 - 어파인 변환과 알파 블렌딩 기법 적용

## 컴파일 및 실행 방법

### 환경 요구사항
- OpenCV 4.11.0
- C++ 컴파일러(g++ 권장)
- 입력 이미지: "lena.jpg", "stitchingL.jpg", "stitchingR.jpg"

### 이미지 회전 프로그램
소스 파일: rotate_skeleton_v2.cpp

컴파일:
```
g++ -std=c++11 rotate_skeleton_v2.cpp -o rotate `pkg-config --cflags --libs opencv4`
```

실행:
```
./rotate
```

### 이미지 스티칭 프로그램 
소스 파일: stitching.cpp

컴파일:
```
g++ -std=c++11 stitching.cpp -o stitch `pkg-config --cflags --libs opencv4`
```

실행:
```
./stitch
```

## 코드 설명 및 사용법

### rotate_skeleton_v2.cpp
이 프로그램은 lena.jpg 이미지를 읽어서 지정된 각도로 회전. 
회전 시 생기는 빈 픽셀을 채우기 위해 두 가지 보간법 구현.

* 입력 파라미터:
  - 회전 각도: 기본값 45도 
    (main 함수에서 `rotated = myrotate<Vec3b>(input, 45, "bilinear")` 부분 수정)
  - 보간법 종류: "nearest" 또는 "bilinear" 
    (main 함수에서 세 번째 인자 수정)

* 특이사항:
  - Nearest Neighbor는 빠르지만 계단 현상 발생 가능
  - Bilinear는 좀 더 부드러운 결과 제공

### stitching.cpp
이 프로그램은 두 개의 이미지(stitchingL.jpg, stitchingR.jpg)를 이어붙여 
파노라마 이미지 생성.

* 주요 기능:
  - 28개의 대응점을 이용한 어파인 변환 행렬 계산
  - 역방향 매핑으로 이미지 워핑
  - 겹치는 부분에 알파 블렌딩 적용

* 조절 가능한 파라미터:
  - 블렌딩 비율(alpha): 
    `blend_stitching(I1, I2, I_f, bound_l, bound_u, 0.5)` 의 마지막 인자
    0에 가까울수록 오른쪽 이미지 특성이 강하고, 
    1에 가까울수록 왼쪽 이미지 특성이 강함.
    기본값은 0.5로 균등하게 블렌딩됨.

## 참고사항
- 두 프로그램 모두 실행 후 아무 키나 누르면 종료.
- 결과 이미지는 화면에 표시되며, 스티칭 결과는 "result.png"로 저장.
- 이미지 회전 시 회전 각도를 크게 하면 결과 이미지가 잘릴 가능성.
- Readme 파일 작성자: 컴퓨터학과 홍길동
- 제출일: 2025년 3월 24일