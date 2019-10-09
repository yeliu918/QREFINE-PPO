# QREFINE-PPO
This is implementation of QREFINE model described in "Generative Question Refinement with Deep Reinforcement Learning in Retrieval-based QA System" (https://arxiv.org/abs/1908.05604)
## Datasets:
* Yahoo dataset:
```
https://ciir.cs.umass.edu/downloads/nfL6/
```
* Customer Service Userlog (CSU):
```
../data/Huawei/large_unique0.95_without_alluserlog
```
## Rewards network:
* Wording Reward:
```
BA_Seq2Seq.py
```
*	Answer Coherency Reward:
```
QA_similiarity.py
```
## Baselines:
*	Seq2Seq and Seq2Seq+ Attention and Seq2Seq + Bidirectional:
```
BA_Seq2Seq.py
```
*	QREFINE_sen:
```
Sen_seqRL_training.py
```
*	QREFINE_word:
```
Word_seqRL_training.py
```
*	QREFINE:
```
Whole_seqRL_training.py
```
## Question Answer Retrieval experiment: 
```
Retrieve_qa.py
```
## Evaluation:
```
Bleu-1-2-3-4: Evaluation_Package/bleu
Meteor: Evaluation_Package/meteor
Rouge: Evaluation_Package/rouge
```
