# RL_Note
Tensorflow 2.4.0 기준으로 작동 확인했습니다.
현재 SAC + HER 공부중입니다.
개인 취미로 RNN + RL 을 해보고 싶습니다.

위 저장소는 다음을 참고했습니다.<br>

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

This repo is refered from below repos<br>

## 파일 구조 설명

notebook<br>
jupyter notebook 파일로 되어있음. pys/utils 의 utility 파일을 사용한다.<br>
대부분 pys를 개발하기 전에 prototyping을 위해서 jupyter notebook을 이용하여 개발하는 곳이다.

pys : python 파일로 되어있음.<br>
discrete_gym.py/ipynb과 continuous_gym.py/ipynb 을 실행하면 gym의 classic control 문제에 대해서 여러 강화학습 알고리즘을 적용해볼 수 있다.<br>
*.py 파일은 강화학습 알고리즘이나 Experience replay 알고리즘에 대해서 Batch Execution을 통해 알고리즘에 대한 성능 비교를 할 수 있도록 하는 batch 파일이다.
- agent <br>
강화학습 알고리즘 모음
- model <br>
강화학습을 위한 심층신경망 클래스로 구성<br>
사용자가 cfg파일을 통해서 심층신경망의 레이어 구조를 결정할 수 있음.
- utils<br>
python utility 파일로 구성. memory, noise 관련한 모듈<br>
현재는 ER, PER, HER이 있으며, HER는 현재 수정 중
- gyms <br>
HER를 위해, done과 reward를 계산하는 함수를 넣음
- result <br>
환경, 강화학습 알고리즘을 선택하여 구동한 결과물을 모았다.<br>
이는 결과 데이터인 data, 결과 이미지인 img, 학습된 신경망 save_model 이 있다.
