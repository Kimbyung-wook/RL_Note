# RL_Note
Tensorflow 2.4.0 기준으로 작동 확인했습니다.
현재 SAC + HER 공부중입니다.
개인 취미로 RNN + RL 을 해보고 싶습니다.

## My file system

* \notebook<br>
There are executable jupyter notebook files they have all of the elements or are refered from \pys\*** <br>
This directory is for proto-typing RL algorithms and neural network model.<br>
실행 가능한 주파이터 노트북 파일이 있습니다. 파일들은 구성요소를 다 가지고 있거나 \pys\*** 등에 있는 파일들을 참조할 수 있습니다.<br>
이 디렉토리는 강화학습 알고리즘과 신경망 모델을 미리 시험해보기 위해 있습니다.

* \pys<br>
There are python scripts about RL<br>
RL consists of RL algorithm, neural network model, hyper-parameters etc...<br>
There are executable jupyter notebook files for testing these module : discrete_gym.ipynb & continuous_gym.ipynb
But it may not be possible to run it. because this repo is under development.<br>
Recently, i made neural network model maker(NNMM), which is not totally tested. Testing of NNMM is been able in the file, the configuration file is at \pys\config<br>
강화학습 관련 파이썬 스크립트로 구성되어있습니다.<br>
 강화학습이란게 강화학습 알고리즘, 신경망 모델, 초-매개변수 등으로 구성됩니다. 이러한 모듈들을 검증하기 위해, 실행 가능한 주파이터 노트푹 파일들이 있습니다. 근데 안 돌아갈 수도 있는데 전체적으로 개발 중이라서 그렇습니다. 최근에는 설정 파일을 읽어서 원하는 합성곱, 혹은 다중 퍼셉트론 레이어를 가지는 신경망 모델을 만들어주는 함수(network_maker.py)를 만들었는데 다 테스트는 못해봤네요. 신경망 생성 함수의 검증은 해당 파일에서, 혹은 \pys\config의 모델 설정 파일을 읽어서 만들 수 있습니다.

* \pys\agent\ <br>
There are the implementation of the RL algorithms<br>
The agent is refered from neural network model(\pys\model\network_maker.py)<br>
이건 강화학습 알고리즘의 구현입니다. 이는 신경망 모델을 참조하고 있습니다.

* \pys\model\ <br>
There are neural network maker<br>
Originally, there were pre-defined neural network models. but it seems that that way to generate NN model is not efficiency, so i made a NN model maker.<br>
여기는 신경망 모델을 만들어주는 함수가 있습니다.<br>
원래 미리 정해진 신경망 모델이 있었는데, 이러한 신경망 생성 방법은 비효율적이라고 생각했습니다. 그래서 신경망 생성기를 만들었답니다?<br>
특정 생성 문법을 지켜준다면 Q-network나 Actor-Critic network를 만들기가 수월하다.

* \pys\utils\ <br>
There are the utility files for RL (ER, PER, HER)<br>
여기는 강화학습의 보조함수를 넣어두었다.
<br>
<br>
<br>
## Reference

위 저장소는 다음을 참고했습니다.<br>
This repo is refered from below repos<br>

* General

* DQN <br>
https://github.com/yc930401/Actor-Critic-pytorch/blob/master/Actor-Critic.py <br>
https://github.com/bentrevett/pytorch-rl/blob/master/2%20-%20Actor%20Critic%20%5BCartPole%5D.ipynb <br>

* DDPG <br>
https://github.com/rlcode/reinforcement-learning-kr-v2<br>
https://keras.io/examples/rl/ddpg_pendulum/ <br>
https://github.com/InSpaceAI/RL-Zoo/blob/master/DDPG.py<br>
https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process<br>
https://keras.io/examples/rl/ddpg_pendulum/ <br>
https://horomary.hatenablog.com/entry/2020/06/26/003806 <br>
https://pasus.tistory.com/138 <br>

* SAC <br>
https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/model.py<br>

* PER <br>
https://ropiens.tistory.com/86 <br>
https://github.com/rlcode/per <br>
https://github.com/takoika/PrioritizedExperienceReplay/blob/master/proportional.py <br>

* HER <br>

* Actor-Critic Series <br>

* Recurrent NN + RL <br>
https://github.com/AntoineTheb/RNN-RL