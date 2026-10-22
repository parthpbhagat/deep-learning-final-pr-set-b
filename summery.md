# Deep Learning & NLP Practical Portfolio: Summary Report

## 1. Five Tasks and Datasets Used
Across this portfolio, five fundamental Deep Learning and NLP tasks were implemented:
1. **ANN Tabular Classification:** Built a multi-layer Artificial Neural Network using the **Wisconsin Breast Cancer Dataset** (569 samples, 30 continuous clinical features) with Dense layers, ReLU, and Dropout to classify tumors as Benign or Malignant.
2. **Computer Vision Preprocessing (OpenCV):** Developed an automated digital image processing pipeline across a multi-object collection (5 images: book, leaf, fruit, mobile phone, product), applying resizing, median filtering, Otsu’s thresholding, contour area measurement, and geometric transformations.
3. **CNN Image Classification:** Trained a Convolutional Neural Network on the **Fashion-MNIST Dataset** (70,000 $28\times28$ grayscale images, 10 apparel classes) with Conv2D filters, MaxPooling downsampling, and Softmax classification.
4. **Sequential Sentiment Analysis (LSTM):** Implemented an Embedding-LSTM architecture on the **IMDB Movie Reviews Dataset** (25,000 polar reviews, 10,000-word vocabulary, 100-token padded sequences) for sentiment classification.
5. **Modern Transformers & NER:** Deployed pre-trained Hugging Face models using **cardiffnlp/twitter-roberta-base-sentiment-latest** on 20 custom product reviews, analyzed WordPiece subword tokenization, and extracted named entities using **dslim/bert-base-NER** across 5 multi-entity sentences.

---

## 2. New Deep Learning Concepts & Learnings
The most valuable new concepts were **Subword Tokenization (WordPiece / BPE)** and the **Self-Attention Mechanism** in Transformers. 

Previously in LSTMs, vocabulary handling relied on fixed word indices, which failed on unknown words. Subword tokenization revealed how language models decompose rare or compound words into frequent subword units marked with `##` prefixes (e.g., splitting `"headphone"` into `"head"` and `"##phone"`), completely eliminating Out-of-Vocabulary (OOV) errors. 

Furthermore, studying Self-Attention demonstrated how modern Transformers transcend recurrent limitations. Instead of sequentially passing hidden states step-by-step through time, Self-Attention computes pairwise token affinities across the whole sentence simultaneously using Query, Key, and Value projections ($\text{softmax}(QK^T / \sqrt{d_k})V$). This allows the model to capture long-range contextual relationships directly without vanishing gradients or sequential bottlenecks.

---

## 3. Results Achieved Across Tasks
- **Task 1 (ANN):** Achieved **~98% test accuracy** with high precision and recall on Breast Cancer classification.
- **Task 2 (OpenCV):** Successfully generated a $5\times5$ preprocessing grid and computed exact contour areas and orientation transforms.
- **Task 3 (CNN):** Achieved **90.8% test accuracy** on Fashion-MNIST with strong diagonal confusion matrix separation across all 10 categories.
- **Task 4 (LSTM):** Reached **86.2% test accuracy** on IMDB reviews, retaining sequential sentiment cues through gated memory cells.
- **Task 5 (Transformers):** Delivered **100% accuracy (16/16)** on clear reviews with $>95\%$ confidence, while confidence dropped to $50\% - 70\%$ on ambiguous reviews, accurately reflecting semantic uncertainty. The BERT NER pipeline extracted person, location, and organization entities with $99\%$ confidence.

---

## 4. What to Build Next (Future Work)
Given additional compute and time, I would explore:
1. **Multimodal Vision-Language Models:** Combine visual encoders (Vision Transformers / CLIP) with language decoders to build automated medical diagnostic systems that jointly analyze histopathology images and clinical notes.
2. **Parameter-Efficient Fine-Tuning (LoRA):** Fine-tune open-source LLMs (Llama 3 / Mistral) on code-mixed Indian-English and Hinglish conversational datasets to resolve current domain limitations in regional e-commerce sentiment and summarization.
