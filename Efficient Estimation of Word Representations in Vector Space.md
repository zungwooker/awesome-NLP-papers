1. **High-level problem:**
   The paper addresses the problem of learning continuous vector representations (embeddings) of words efficiently from large datasets. These representations capture syntactic and semantic word similarities, which can improve natural language processing (NLP) tasks such as speech recognition, machine translation, and question answering.

2. **Limitations of previous research:**
   The authors critique existing models, particularly those that rely on N-gram language models or neural network-based models. They argue that while these models perform well, they are computationally expensive and inefficient for large datasets. Most previous methods could not scale beyond hundreds of millions of words or low-dimensional word vectors, limiting their accuracy and practical application in real-world tasks.

3. **Motivation for the proposed method:**
   The motivation behind the authors' methods is to develop a more computationally efficient approach that can scale to much larger datasets (billions of words) while maintaining or improving the quality of the learned word vectors. They aim to minimize the computational complexity, particularly by reducing the need for non-linear hidden layers, which are the primary source of computational cost in previous methods.

4. **Contribution of the proposed method:**
   The paper introduces two novel architectures: the **Continuous Bag-of-Words (CBOW) model** and the **Skip-gram model**. Both are designed to efficiently learn word vectors that capture syntactic and semantic relationships. The CBOW model predicts a word based on its context, while the Skip-gram model predicts context words from a given word. These models reduce training complexity while maintaining high accuracy in word similarity tasks.

5. **Role of different components in the contribution:**
   - **CBOW model:** It improves efficiency by removing the non-linear hidden layer, using a shared projection layer where word vectors are averaged. This simplification significantly reduces training time while providing good performance in syntactic tasks.
   - **Skip-gram model:** It predicts context words from a given word within a sentence. By adjusting the range of context words considered, the model can balance between accuracy and computational complexity. This model performs better on semantic tasks and is especially effective when trained on large datasets.
---

1. **해결하고자 하는 문제:**  
   이 논문은 대규모 데이터셋에서 단어의 연속적 벡터 표현(임베딩)을 효율적으로 학습하는 문제를 다룹니다. 이러한 벡터 표현은 단어 간의 구문적, 의미적 유사성을 포착하여 음성 인식, 기계 번역, 질의 응답 시스템과 같은 자연어 처리(NLP) 작업의 성능을 향상시킬 수 있습니다.

2. **이전 연구의 한계:**  
   저자들은 기존의 N-그램 언어 모델이나 신경망 기반 모델이 성능은 좋지만, 대규모 데이터셋에서 매우 비효율적이며 계산 비용이 많이 든다고 지적합니다. 이전 방법들은 수억 개의 단어 또는 낮은 차원의 단어 벡터를 학습하는 데 그쳤으며, 이는 정확도와 실제 적용에 한계를 가져옵니다.

3. **제안된 방법의 동기:**  
   저자들은 더 큰 데이터셋(수십억 개의 단어)을 효율적으로 처리하면서도 학습된 단어 벡터의 품질을 유지하거나 개선할 수 있는 방법을 개발하고자 합니다. 이들은 특히 신경망의 비선형 은닉층을 제거하여 계산 복잡도를 줄이는 데 중점을 둡니다. 이는 계산 비용을 낮추는 주요한 방법입니다.

4. **제안된 방법의 기여:**  
   논문에서는 두 가지 새로운 아키텍처, **Continuous Bag-of-Words(CBOW) 모델**과 **Skip-gram 모델**을 제안합니다. 이 모델들은 효율적으로 단어 벡터를 학습하여 구문적, 의미적 관계를 잘 포착합니다. CBOW는 문맥을 기반으로 단어를 예측하고, Skip-gram은 주어진 단어를 사용하여 문맥의 단어들을 예측합니다. 이러한 모델들은 계산 복잡도를 낮추면서도 높은 정확도를 유지합니다.

5. **각 구성 요소의 역할:**  
   - **CBOW 모델:** 비선형 은닉층을 제거하고 투영층을 공유하여 단어 벡터를 평균화하는 방식으로 효율성을 크게 개선하였습니다. 이 단순화는 학습 시간을 크게 단축하면서도 구문적 과제에서 좋은 성능을 발휘합니다.
   - **Skip-gram 모델:** 문장 내에서 주어진 단어를 기준으로 문맥 단어들을 예측하는 방식으로, 문맥의 범위를 조정하여 정확도와 계산 복잡도를 균형 있게 유지합니다. 이 모델은 특히 대규모 데이터에서 의미적 작업에서 더 뛰어난 성능을 보입니다.
---

### 1. N-gram 언어 모델

N-gram 언어 모델은 **단어들의 연속적인 n개의 묶음(N-gram)**을 기반으로 다음에 나올 단어의 확률을 예측하는 모델입니다. 여기서 **n**은 연속적으로 관찰되는 단어의 수를 나타냅니다.

예를 들어, 3-gram(혹은 트리그램) 모델에서는 세 개의 연속된 단어로 다음 단어를 예측합니다. 문장이 “I love machine learning”이라면, "I love machine"까지 주어졌을 때 "learning"이 나올 확률을 계산하는 방식입니다.

N-gram 모델은 상대적으로 단순하고 직관적이며, 데이터의 빈도 기반으로 학습이 가능합니다. 다만, 연속적인 단어에만 집중하다 보니 **긴 문맥을 고려하지 못하고**, **대규모 데이터**에서는 높은 차원의 희소성 문제가 발생할 수 있습니다. 새로운 단어 조합을 만났을 때는 예측이 어렵다는 한계도 있습니다.

### 2. Skip-gram의 장점

**Skip-gram** 모델의 주요 장점은 다음과 같습니다:

1. **멀리 있는 단어까지 학습 가능**: Skip-gram은 주어진 단어를 기준으로 주변 문맥의 단어들을 예측합니다. 특히, 문장에서 **멀리 떨어진 단어까지 학습**할 수 있어, 문맥의 범위를 넓게 포착할 수 있습니다.
   
2. **단어 관계 학습**: Skip-gram은 한 단어가 여러 문맥에서 어떻게 사용되는지를 학습하여 단어 간의 구문적, 의미적 유사성을 잘 포착할 수 있습니다. 이를 통해 **단어 벡터 사이의 관계**(예: "king" - "man" + "woman" = "queen")를 자연스럽게 학습할 수 있습니다.

3. **드문 단어에 유리**: Skip-gram은 자주 등장하지 않는 **드문 단어**들의 벡터 표현을 더 잘 학습합니다. 이는 주어진 단어를 문맥과 함께 학습하면서, 단어의 드문 사용 패턴도 효율적으로 포착하기 때문입니다.

### Skip-gram의 장점이 어디에서 오는가?

- **단순한 아키텍처**: Skip-gram은 CBOW와 다르게, 중간에 **비선형 은닉층 없이** 바로 단어와 단어 간의 관계를 학습합니다. 이로 인해 계산 복잡도는 낮추고, 학습 속도는 향상됩니다.

- **문맥의 폭**: Skip-gram은 주어진 단어 주변의 **넓은 범위의 문맥 단어들**을 예측하는데, 이때 문맥 단어들의 거리가 멀어질수록 가중치를 줄이는 방식으로 계산 복잡도를 줄이면서도 장거리 문맥 정보를 활용합니다. 이를 통해 더 정교한 단어 벡터를 학습할 수 있습니다.
