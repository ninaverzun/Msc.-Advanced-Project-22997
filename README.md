# Empowering Conversations: How Machine Learning Can Help Us Tackle Online Hate

This repository contains the results of an Msc.-Advanced-Project in The Open University of Israel (22997) titled "Empowering Conversations: How Machine Learning Can Help Us Tackle Online Hate".

## Counterspeech generation

Instances of Hate Speech sourced from Reddit, taken from:
"A Benchmark Dataset for Learning to Intervene in Online Hate Speech" available here: https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech/blob/master/data/reddit.csv

For genration Llama-3.2-1B model from Meta {https://huggingface.co/meta-llama/Llama-3.2-1B} was used.

---

### 📊 Evaluation of Counter-Speech Prompt Outcomes
|   | **Required Outcome**               | **Prompt**                                                                                                                                                                                                 | **Valid CS (%)** | **Non-Hateful Hater Reentry (%)** | **Low Conversation Incivility (%)** | **METEOR**         | **BERT Score**       | **Llama-3.2-1B Generation Result**                                                                                                                                                                                                 | **Incivility Level Predictions {0: 'barely', 1: 'somewhat', 2: 'highly'}** |**Hater Reentry Predictions {0: 'Non-hateful reentry', 1: 'Hateful reentry', 2: 'No reentry'}** |
|---|-----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|------------------------------------|--------------------------------------|--------------------|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|-------------------------------|
| 1 | **Reference (crowd sourced)**     | -                                                                                                                                                                                                         | -                | 42.8%                             | 24.7%                               | 1                  | 1                    | Human-written baseline                                                                                                                                                                                                             | [Crowd Civility](https://drive.google.com/file/d/1048xQZuY5lwXBcMG2cwC2uWxRsWrho48/view?usp=sharing)      | [Crowd Reentry](https://drive.google.com/file/d/1BnR4ZGI76St-U0T3lOGxnTKb3rPG37qh/view?usp=sharing)|
| 2 | **None** *(baseline)*             | *"You are a helpful and understanding Redditor who aims to de-escalate conflict. Generate a constructive and polite counter-speech reply to the following Reddit comment.*<br>**{Hate Comment}**<br>Counter-speech:" | ~99.85%          | 54.9%                             | 35.7%                               | 0.025 (0.057)      | 0.814 (0.018)        | [Baseline CS](https://drive.google.com/file/d/1hOAMpvrUZTl45Z9jlLVa6dFcJ-PncNCk/view?usp=drive_link), [Generation code part 1](https://colab.research.google.com/drive/1ZKN7wE44mZkf9b3Klk0iPMShtIPyFTnD?usp=sharing), [Generation code part 2](https://colab.research.google.com/drive/1psQJFmbA28XQWeWH6DWrmnjRsagMnFrf?usp=sharing)                                                                                                                          | [Baseline](https://drive.google.com/file/d/1agOUCUihrpLPAAMf2PwUq_3mieEDTXQC/view?usp=sharing)                      |[Baseline Reentry](https://drive.google.com/file/d/1PIJWIVtFxKSmz7oW357b9cPq7XCsNrF_/view?usp=sharing)|
| 3 | **Low conversation incivility**   | *"You are a helpful and understanding Redditor who aims to de-escalate conflict. Generate a constructive and polite counter-speech reply to the following Reddit comment, so that it could lead to low incivility in the following conversations.*<br>**{Hate Comment}**<br>Counter-speech:" | ~99.89%          | 68.2%                             | 29.8%                               | 0.015 (0.035)      | 0.814 (0.020)        | [Civil CS](https://drive.google.com/file/d/1RxAwWhmebFRrrFOBrkGuVgaE54eHHCGK/view?usp=drive_link), [Generation code](https://colab.research.google.com/drive/1aP1in4jBS_ExNppMnUSXtFYnvqeCNR-c?usp=sharing)| [Civil](https://drive.google.com/file/d/1pzNzrWZXalJSpxWZ3DvTyMNPW0klkUYN/view?usp=sharing)                           | [Civility prompt Reentry](https://drive.google.com/file/d/1-0yZ8lD_ObxlD6-axh_cCon7-3kLWabT/view?usp=sharing)|
| 4 | **Non-hateful hater reentry**     | *"You are a helpful and understanding Redditor who aims to de-escalate conflict. Generate a constructive and polite counter-speech reply to the following Reddit comment so that the hater will come back and have constructive engagement in the conversation.*<br>**{Hate Comment}**<br>Counter-speech:" | ~99.79%          | 60.2%                             | 32.8%                               | 0.016 (0.0137)     | 0.824 (0.012)        | [Non-hateful hater reentry CS](https://drive.google.com/file/d/12IpZY9_5mwskmqU3CFVDhI7ULH5tPkn_/view?usp=drive_link), [Generation code part1](https://colab.research.google.com/drive/1m4gIsc8Plc0t7LG-YBjkyDYmLeVPZtXw?usp=sharing), [Generation code part 2](https://colab.research.google.com/drive/1pUWDoNOWtr0shokI2IdiHi8WsWbTWYg2?usp=sharing)                                                                                                                   | [Non-hateful hater reentry prompt Civility](https://drive.google.com/file/d/1PCpckzHY7ca1ESV1OVTCEOaYh9Ivr8jW/view?usp=sharing)          |[Non-hateful hater reentry](https://drive.google.com/file/d/1LkAPegR-uq5ozfvNYjztpoc-G5WISf0y/view?usp=sharing)|
| 5 | **None** *(baseline)*             |**System**<br>”Generate a response in Reddit Style. <br>**User**<br>”Here is the Reddit comment:<br>**{Hate Comment}**<br>. Please write a counterspeech to the Reddit hate comment." | ~73.70%  | -  | -   | 0.025 (0.057)    | 0.814 (0.018)    |   [Baseline Prompt Results](https://drive.google.com/file/d/1-BOAzUALnn6wBEPC9xImeQI39ro2rAai/view?usp=sharing) |||
| 6 | **Low conversation incivility**   |**System**<br>”Generate a response in Reddit Style. <br>**User**<br>”Here is the hate comment:<br>**{Hate Comment}**<br>. Please write a counterspeech to the hate comment so that it could lead to low incivility in the following conversations." | ~59.57%  | -  | -   | 0.015 (0.035)     | 0.814 (0.020)    |  [Civility Prompt Generation Code](https://colab.research.google.com/drive/1f-NrvXderYt7gbv-snwykmg9mf_xUbRq?usp=sharing), [Civility Prompt Results](https://drive.google.com/file/d/1--GX7E3v8k31n-u_JX6UC-V2xUKpomxt/view?usp=sharing)   |       ||
| 7 | **Non-hateful hater reentry**  |**System**<br>”Generate a response in Reddit Style. <br>**User**<br>”Here is the hate comment:<br>**{Hate Comment}**<br>. Please write a counterspeech to the hate comment so that the hater will come back and have constructive engagement in the conversation." | ~62.91%  | -  | -   | 0.016 (0.032)    | 0.816 (0.021)     |   [Reentry Prompt Generation Code](https://colab.research.google.com/drive/1Xvb92ZtPsE4wI9wLGJscNR3E3wKlMVl7?usp=sharing), [Reentry Prompt Results](https://drive.google.com/file/d/14wD7zwljCWRX7jv0XFXi66X3WtEL7EB4/view?usp=sharing)  |       ||

---

## Incivility level classifier 

Incivility level classifier training was done using  the corpus of the ICWSM 2024 paper "Hate Cannot Drive out Hate: Forecasting Conversation Incivility following Replies to Hate Speech"
(Authors: Xinchen Yu, Eduardo Blanco, and Lingzi Hong) available here [Conversation Incivility corpus](https://raw.githubusercontent.com/xinchenyu/incivility/refs/heads/main/data.csv).
Incivility levels mapping: {0: 'barely', 1: 'somewhat', 2: 'highly'}

This notebook was used to train be best classifier: [Incivility best classifier training notebook](https://colab.research.google.com/drive/1u_GcGzhAW090ut2kqSygVZ6r66hGVsoh?usp=sharing). Best Classifier is saved here: [Incivility best classifier model](https://drive.google.com/drive/folders/1gpoSUb7MZVCw7lxfogrBBL5jma-8XaEi?usp=sharing)

The Incivility level predictions were done using this notebook:
[Incivility level predictins with best classifier](https://colab.research.google.com/drive/17afid54vG1uTHjJLcLjUX8rteKcF8mwH?usp=drive_link)

### Incivility level classifier atrchitecture: 

The best model was based on 🧠 CustomRobertaClassifier Architecture. This model defines a **custom classifier based on `roberta-base`**, extending it with additional fully connected layers for downstream multi-class classification.

#### 📦 Class Definition

```python
class CustomRobertaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 768)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(768, 3)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.last_hidden_state[:, 0, :])  # CLS token
        x = self.tanh(self.fc1(x))
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)
```

#### 🧬 Architecture Breakdown

| Component            | Description                                                                 |
| -------------------- | --------------------------------------------------------------------------- |
| `RobertaModel`       | Pretrained `roberta-base` model from HuggingFace Transformers               |
| `Dropout(p=0.1)`     | Dropout applied to the pooled `[CLS]` representation                        |
| `Linear(768 → 768)`  | Fully connected hidden layer for transformation of CLS output               |
| `Tanh()`             | Non-linear activation function after `fc1`                                  |
| `Linear(768 → 3)`    | Final classification layer mapping to 3 output classes                      |
| `CrossEntropyLoss()` | Loss function for training when `labels` are provided (multi-class setting) |

#### 🔁 Forward Pass

1. Input token IDs and attention masks are passed through `roberta-base`.
2. The `[CLS]` token representation is extracted: `outputs.last_hidden_state[:, 0, :]`.
3. Dropout regularization is applied.
4. The result passes through a linear layer + tanh activation.
5. A final linear layer maps to logits for 3-class classification.
6. If labels are provided, a cross-entropy loss is computed.

#### ✅ Output

Returns a HuggingFace `SequenceClassifierOutput` containing:

* `loss`: Cross-entropy loss (if labels provided)
* `logits`: Raw class scores (before softmax)

## Non-hateful hater reentry Classifier Training

Hater reentry behaviour classifier training was conducted using the **ReEco corpus**, introduced in the NAACL 2025 Findings paper:  
**"Echoes of Discord: Forecasting Hater Reactions to Counterspeech"**  
[@song2025echoes](#citation)

The ReEco corpus comprises **5,723 (Hate Speech, Counterspeech)** pairs collected from Reddit, annotated to facilitate analysis of haters' reactions to counterspeech.

📂 **Dataset available here**:  
[Hater Behavior & Counterspeech Reactions CSV](https://github.com/oliveeeee25/counterspeech_effectiveness_hater_reentry/blob/main/hater_behavior_counterspeech.csv)

📄 **Column Descriptions**:
- `grandContent`: Hate speech  
- `parentContent`: Counterspeech  
- `sonContent`: Reply to counterspeech  
- `re_entry`:  
  - `1` → Reentry (Hater responds again after counterspeech)  
  - `0` → No reentry (Hater does not respond)  
- `sonLabel` & `3_category`:  
  - `1` → Hateful reentry  
  - `0` → Non-hateful reentry  
  - `2` → No reentry  

---
This notebook was used to train be best classifier: [Non-hateful hater reentry classification train](https://colab.research.google.com/drive/18S3MBEvlnW-sLma2BmE8h9oPZwco75GV?usp=sharing).
Best hater-reentry behaviour classifier is saved here: [Non-hateful hater reentry best classifier model](https://drive.google.com/drive/folders/1tIHsCr-QWrsHzI4ae1KxRwhyaTxtVreK?usp=sharing).

This noteboot was used for Benchmark data predictions: [Hater-reentry behaviour classification](https://colab.research.google.com/drive/1Pd6rMNwZgtJSiB7LxIlOm1Tai_tcJVJy?usp=sharing).

### Citation

```bibtex
@article{song2025echoes,
  title={Echoes of Discord: Forecasting Hater Reactions to Counterspeech},
  author={Song, Xiaoying and Perez, Sharon Lisseth and Yu, Xinchen and Blanco, Eduardo and Hong, Lingzi},
  journal={arXiv preprint arXiv:2501.16235},
  year={2025}
}



