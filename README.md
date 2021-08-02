# RL_Note
Tensorflow 2.4.0 기준으로 작동 확인했습니다.
현재 SAC + HER 공부중입니다.

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

This repo is refered from below repos<br>

## 파일 구조 설명

notebook : jupyter notebook 파일로 되어있음. pys/utils 의 utility 파일을 사용한다.

pys : python 파일로 되어있음.<br>
pys/utils : python utility 파일로 구성. memory, noise 관련한 모듈<br>
