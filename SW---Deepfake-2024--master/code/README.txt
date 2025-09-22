실행환경: 운영체제 - Ubuntu 22.04.3 LTS
	      Anaconda3 가상환경에서 진행하였음
	      conda 명령어로 가상환경, 커널 생성 후 설치 진행

===================라이브러리 요구 버전======================
python == 3.11.7

- mixing_voice.ipynb
pip install pandas==2.2.2
pip install tqdm==4.66.4

- CQT.ipynb
pip install librosa==0.10.2.post1
numpy == 2.0.1

- denoised.py
torch == 2.3.1
soundfile ==0.12.1
pip install torchaudio==2.3.1
pip install deepfilternet
※ deepfilternet 설치 중 오류가 뜰 시 
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
pip install deepfilternet

- yamnet.ipynb
pip install tensorflow==2.17.0
pip install resampy==0.4.3
pip install tensorflow_hub==0.16.1

========================================================



=======================실행 순서 설명========================

@@ 필요한 모든 경로는 상대경로로 지정했으므로 경로 수정을 따로 하실 필요가 없습니다! 단, 검증 시 필요한 train , test 데이터셋의 경로는 지정해 주셔야 합니다 !! @@

1. 데이터 증강을 위해 받은 데이터에서 real과 fake를 합친 데이터 생성함. 
    mixing_voice.ipynb 파일에 있는 경로는 따로 지정해 줄 필요가 없음
    train.csv 파일 불러온 후 real은 real끼리, fake는 fake끼리 묶음
    real인 오디오와 fake인 오디오를 각각 하나로 합침 .
    train.csv 의 label을 각각 real, fake 열로 만들어 1과 0으로 나타냄
    real과 fake가 합쳐진 오디오는 real도 1, fake도 1로 설정
    
   실행 후 train_mixed.csv가 생성됨
   
   mixed 된 파일들은 mixed 디렉토리 내에 저장되고, mixed 내에 있는 모든 파일은 train 폴더로 mv 시켜주면 됨
   ( # (1), # (2) 주석 참고)
   ★ mixed 디렉토리로 이동 후, mv * ../train 명령어 실행 ★

2. 전처리가 필요한 train데이터와 test데이터를 CQT코드에 각각 넣어서 실행 한 후 npy 파일로 변환 
	2-1. 결과를 저장할 디렉토리의 경로를 지정해준 후 첫번째 슬롯 실행 후 두번 째 슬롯 실행
	2-2. 그 후 max_value값을 본 후 4번째 슬롯에 max_value값 넣어준 후 실행

3. model_train 폴더에 있는 multigpu.py코드에 있는 49, 170, 229 라인에 있는 경로의 파일명은 매번 실행마다 다르게 적어야 코드 컴파일 가능
	3-1. 36라인에서 학습률(Learning_rate) 수정
	3-2. 231,232 라인에 마지막 부분에 (2)에서 CQT전처리한 npy파일 폴더의 경로를 두군데 다 넣어주고 , 그 앞부분에 있는 경로는 
        1번 에서 만든 train_mixed.csv 파일을 랜덤으로 80대 20으로 train데이터와 valid 데이터로 나눈 csv 경로임 그 후 
	torchrun --nnode=1 --node_rank=0 --nproc_per_node=4 multigpu.py 1000 --batch_size=16
	위 명령어 친 후 실행

위 과정을 거쳐 만들어진 가중치 파일이 model_weight 폴더안에 weight_mixdrop10.pt 파일임.

만들어진 모델로 test를 진행 전 

1. test데이터를 denoised 전처리를 하기위해 denoised.py파일을 켜서 test를 하기위한 ogg 파일이 있는 폴더의 경로와 
소음 제거된 파일을 저장할 디렉토리 경로를 설정해야함 그 후 -> python denoised.py 명령어를 터미널에 입력해 컴파일 시작

2. denoised 전처리된 파일은 wav 형태로 저장이 됨

3. yamnet모델로 각 음성파일별 사람이 존재할 확률 추출
	yamnet.ipynb 에 있는 코드를 실행
	# (1) 에는 소음 제거된 파일이 저장된 디렉토리 경로를 지정해야 함
	이 코드에서 만들어진 voice_denoised_yamnet.csv 파일에 경로를 추가하고 파일명을 오름차순으로 정렬한 후
	model_start 폴더의 updated_test_origin_cqt.csv 파일로 저장하였음

=====================model_start 폴더 진행=======================

1. model_start 폴더에 있는 SSDNET3.py 파일을 열어서 10번 라인에 model_weight폴더에 있는 가중치 경로를 설정, 
16번 라인에 위에서 만든 yamnet을 돌려 나온 확률과 CQT 전처리한 test데이터의 경로가 있는 csv파일 경로를 지정 (model_start안에 있는 updated_test_origin_cqt.csv 파일)

2. SSDNET3.py 실행 ==> 제출할 파일 생성 

