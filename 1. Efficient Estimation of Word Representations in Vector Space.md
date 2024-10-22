1. **High-level problem:**
   The paper addresses the problem of learning continuous vector representations (embeddings) of words efficiently from large datasets. These representations capture syntactic and semantic word similarities, improving natural language processing (NLP) tasks such as speech recognition, machine translation, and question answering.

2. **Limitations of previous research:**
   The authors critique existing models, particularly those that rely on N-gram language models or neural network-based models. They argue that while these models perform well, they are computationally expensive and inefficient for large datasets. Most previous methods could not scale beyond hundreds of millions of words or low-dimensional word vectors, limiting their accuracy and practical application in real-world tasks.

3. **Motivation for the proposed method:**
   The motivation behind the authors' methods is to develop a more computationally efficient approach that can scale to much larger datasets (billions of words) while maintaining or improving the quality of the learned word vectors. They aim to minimize the computational complexity, particularly by simplifying the learning process without the need for complex architectures.

4. **Contribution of the proposed method:**
   The paper introduces two novel architectures: the **Continuous Bag-of-Words (CBOW) model** and the **Skip-gram model**. Both are designed to efficiently learn word vectors that capture syntactic and semantic relationships. The CBOW model predicts a word based on its context, while the Skip-gram model predicts context words from a given word. These models reduce training complexity while maintaining high accuracy in word similarity tasks.

5. **Role of different components in the contribution:**
   - **CBOW model:** It improves efficiency by using a shared projection layer where word vectors are averaged. This simplification significantly reduces training time while providing good performance in syntactic tasks.
   - **Skip-gram model:** It predicts context words from a given word within a sentence. By adjusting the range of context words considered, the model can balance between accuracy and computational complexity. This model performs better on semantic tasks and is especially effective when trained on large datasets.
  
---

1. **해결하고자 하는 문제:**  
   이 논문은 대규모 데이터셋에서 단어의 연속적 벡터 표현(임베딩)을 효율적으로 학습하는 문제를 다룹니다. 이러한 벡터 표현은 단어 간의 구문적, 의미적 유사성을 포착하여 음성 인식, 기계 번역, 질의 응답 시스템과 같은 자연어 처리(NLP) 작업의 성능을 향상시킬 수 있습니다.

2. **이전 연구의 한계:**  
   저자들은 기존의 N-그램 언어 모델이나 신경망 기반 모델이 성능은 좋지만, 대규모 데이터셋에서 매우 비효율적이며 계산 비용이 많이 든다고 지적합니다. 이전 방법들은 수억 개의 단어 또는 낮은 차원의 단어 벡터를 학습하는 데 그쳤으며, 이는 정확도와 실제 적용에 한계를 가져옵니다.

3. **제안된 방법의 동기:**  
   저자들은 더 큰 데이터셋(수십억 개의 단어)을 효율적으로 처리하면서도 학습된 단어 벡터의 품질을 유지하거나 개선할 수 있는 방법을 개발하고자 합니다. 이들은 학습 과정의 복잡한 아키텍처 없이 계산 복잡도를 줄이면서도 성능을 유지하는 데 중점을 두었습니다.

4. **제안된 방법의 기여:**  
   논문에서는 두 가지 새로운 아키텍처, **Continuous Bag-of-Words(CBOW) 모델**과 **Skip-gram 모델**을 제안합니다. 이 모델들은 효율적으로 단어 벡터를 학습하여 구문적, 의미적 관계를 잘 포착합니다. CBOW는 문맥을 기반으로 단어를 예측하고, Skip-gram은 주어진 단어를 사용하여 문맥의 단어들을 예측합니다. 이러한 모델들은 계산 복잡도를 낮추면서도 높은 정확도를 유지합니다.

5. **각 구성 요소의 역할:**  
   - **CBOW 모델:** 투영층을 공유하여 단어 벡터를 평균화하는 방식으로 효율성을 크게 개선하였습니다. 이 단순화는 학습 시간을 크게 단축하면서도 구문적 과제에서 좋은 성능을 발휘합니다. 작은 데이터셋에 대해서 Skip-gram 대비 더 잘 동작하며, 빈번하게 나타나는 단어에 대해 더 잘 임베딩하는 장점이 있음.
   - **Skip-gram 모델:** 문장 내에서 주어진 단어를 기준으로 문맥 단어들을 예측하는 방식으로, 문맥의 범위를 조정하여 정확도와 계산 복잡도를 균형 있게 유지합니다. 이 모델은 특히 대규모 데이터에서 의미적 작업에서 더 뛰어난 성능을 보입니다. 희소한 단어에 대해 잘 표현하는 장점이 있음.
