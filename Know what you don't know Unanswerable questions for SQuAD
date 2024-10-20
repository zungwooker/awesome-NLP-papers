### 1. **High-Level Problem**:
The paper introduces **SQuAD 2.0 (Stanford Question Answering Dataset 2.0)**, a dataset designed to challenge reading comprehension models by including **unanswerable questions**. While many models can locate correct answers from a context document, they often fail when presented with questions where the correct answer is not in the provided text. The goal of SQuAD 2.0 is to evaluate models on both answering questions when possible and **identifying when no answer is available** in the context, ensuring models learn to “know what they don’t know.”

### 2. **Limitations of Previous Research**:
Prior datasets, including **SQuAD 1.1**, focused on **answerable questions** only. Models trained on these datasets became good at selecting spans in a document related to the question but lacked the ability to abstain from answering when the question had no valid answer. Some datasets with **unanswerable questions** were automatically generated, making them too simplistic and easily identifiable. These datasets failed to sufficiently challenge models, as they relied on simple heuristics such as **type-matching** or **keyword overlap** to distinguish answerable from unanswerable questions.

### 3. **Motivation for the Proposed Method**:
The motivation behind SQuAD 2.0 is to create a more realistic and challenging reading comprehension task by incorporating **human-crafted unanswerable questions** that closely resemble answerable ones. This encourages models to go beyond simple heuristics and develop a deeper understanding of the text. By incorporating over **50,000 unanswerable questions**, the new dataset pushes models to determine when they should abstain from answering, simulating real-world scenarios where no valid answer may exist in the context.

### 4. **Methodology**:
The methodology for creating SQuAD 2.0 involves the following steps:
- **Dataset Composition**: SQuAD 2.0 combines the original SQuAD 1.1 dataset (which contains over 100,000 answerable questions) with **53,775 unanswerable questions**. These unanswerable questions were written adversarially by crowdworkers to look like answerable ones.
- **Question Design**: Crowdworkers were tasked with creating questions that are relevant to the text but have no answer within the provided paragraph. These questions were designed to have plausible answers, i.e., spans in the text that seem to be valid but are incorrect.
- **Training and Evaluation**: Models are trained to predict both answer spans and the probability that a question is unanswerable. At evaluation, models are penalized for providing answers to unanswerable questions and rewarded for correctly abstaining.

### 5. **Contribution of the Method**:
- **Introducing Unanswerable Questions**: SQuAD 2.0 introduces a significant challenge for models by incorporating **unanswerable questions** into the dataset. This enhances the evaluation of a model’s **robustness** and **reasoning ability**, as it forces models to determine when no answer exists in the text.
- **Harder than SQuAD 1.1**: A state-of-the-art model that achieved **86% F1** on SQuAD 1.1 scored only **66% F1** on SQuAD 2.0, demonstrating the increased difficulty of the task.
- **Diagnostic Insights**: The paper highlights that current models are particularly prone to being fooled by **plausible but incorrect answers**, showing a gap in their ability to understand and reason about the context.

### 6. **Role of Different Components**:
- **Adversarial Unanswerable Questions**: These questions are specifically designed to confuse models, presenting plausible distractors. Models that rely on superficial text matching, such as keyword overlap, are likely to fail on these.
- **Evaluation Metrics**: The exact match (EM) and F1 scores used in SQuAD 2.0 reward models for identifying unanswerable questions and abstaining from giving incorrect answers. This shifts the focus from merely finding answers to assessing comprehension and reasoning.
- **Model Baselines**: Three model architectures were tested: **BiDAF**, **DocQA**, and **DocQA with ELMo**. The best model, DocQA + ELMo, achieved a score of **66.3% F1** on the SQuAD 2.0 test set, compared to human performance at **89.5% F1**, illustrating the complexity of the task.

---

### 1. **해결하고자 하는 문제**:
이 논문은 **SQuAD 2.0(스탠포드 질문 응답 데이터셋 2.0)**을 소개하며, 읽기 이해 모델이 **답변할 수 없는 질문**을 처리하는 능력을 평가하는 것을 목표로 합니다. 기존 모델들은 텍스트 내에서 정답을 찾는 데는 능숙했지만, 문맥에서 정답이 없는 경우에도 무작위로 추측하는 경향이 있었습니다. SQuAD 2.0은 모델이 답변이 가능한지 여부를 결정하는 능력도 평가하여, 모델이 모르는 것을 '모른다'고 판단할 수 있도록 하는 도전 과제를 제시합니다.

### 2. **이전 연구의 한계**:
이전 데이터셋, 특히 **SQuAD 1.1**은 **답변 가능한 질문**만을 포함하고 있었기 때문에, 모델이 단순히 문맥과 질문 간의 연관성을 기반으로 정답을 추측하는 데 그쳤습니다. 또한, 일부 **답변 불가능한 질문**을 자동으로 생성한 데이터셋은 지나치게 단순해, 모델이 키워드 일치나 유형 일치를 통해 쉽게 구분할 수 있었습니다. 이러한 데이터셋은 모델을 충분히 도전시키지 못했습니다.

### 3. **제안된 방법의 동기**:
SQuAD 2.0의 동기는 더 현실적이고 도전적인 읽기 이해 과제를 만들기 위해 **사람이 작성한 답변 불가능한 질문**을 포함하는 것입니다. 이로 인해 모델은 단순한 규칙을 넘어서 문맥을 깊이 이해하고 판단해야 합니다. **50,000개 이상의 답변 불가능한 질문**이 추가됨으로써, 모델은 언제 답변을 포기해야 하는지를 학습하게 됩니다.

### 4. **방법론**:
SQuAD 2.0을 구성하는 방법은 다음과 같습니다:
- **데이터셋 구성**: SQuAD 2.0은 원래 SQuAD 1.1 데이터(100,000개 이상의 답변 가능한 질문)에 **53,775개의 답변 불가능한 질문**을 추가하여 구성되었습니다. 이러한 질문들은 사람들에 의해 작성되었으며, 답변 가능한 질문과 매우 유사하게 보이도록 설계되었습니다.
- **질문 디자인**: 크라우드 작업자들은 문맥과 관련 있지만 문단 내에서는 답변할 수 없는 질문을 작성하도록 요구되었습니다. 이러한 질문은 타당해 보이는 답변을 가지고 있어야 하며, 이는 잘못된 답변을 유도할 수 있는 방식으로 설계되었습니다.
- **훈련 및 평가**: 모델은 답변을 추측하는 것뿐만 아니라, 질문이 답변이 불가능한 경우에 **정답을 포기하는 확률**도 학습하게 됩니다. 평가 시, 답변할 수 없는 질문에 답변을 시도하면 페널티를 받고, 이를 잘 구분해낼 경우 보상을 받습니다.

### 5. **제안된 방법의 기여**:
- **답변 불가능한 질문 도입**: SQuAD 2.0은 모델이 단순한 답변 추출을 넘어 문맥을 **깊이 이해**하고, 없는 답을 찾으려는 시도를 피할 수 있도록 하는 새로운 도전 과제를 제시합니다.
- **SQuAD 1.1보다 어려운 데이터셋**: 기존 SQuAD 1.1에서 **86% F1**을 기록한 최첨단 모델은 SQuAD 2.0에서 **66% F1**로 성능이 크게 떨어졌습니다. 이는 SQuAD 2.0의 난이도가 훨씬 더 높음을 보여줍니다.
- **진단 통찰**: 이 논문은 모델이 **타당하지만 잘못된 답변**에 쉽게 속을 수 있음을 강조하며, 현재 모델들이 문맥을 완벽하게 이해하는 데 있어 한계를 가지고 있음을 보여줍니다.

#### 6. **각 구성 요소의 역할**:
- **적대적인 답변 불가능한 질문**: 이러한 질문은 모델이 쉽게 속을 수 있는 방식으로 설계되어, 단순한 키워드 일치에 의존하는 모델이 실패하도록 유도합니다.
- **평가 메트릭**: 정확한 일치(EM)와 F1 점수는 모델이 답변 불가능한 질문을 정확히 구분하고, 잘못된 답변을 하지 않도록 평가하는 데 중점을 둡니다.
- **모델 성능 기준**: 세 가지 모델이 테스트되었으며, **BiDAF**, **DocQA**, **DocQA with ELMo** 중 DocQA + ELMo가 **66.3% F1**으로 가장 높은 성능을 기록했습니다.
