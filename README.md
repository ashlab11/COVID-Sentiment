# COVID-Sentiment
Final Project + Paper for COMP9444 @ UNSW

Important Files:
- DistilBERT fine-tuning can be found in [this](BERT.ipynb) file. I tried to precompute hidden states in [this](BERT_Precompute.ipynb), but the memory requirements overran my macbook.
- The trained transformer can be found in [this](Trained_Transformer.ipynb) file. This was trained completely from scratch, and uses 8 total attention mechanisms. It performs fine, but definitely doesn't have the data needed to perform well enough.
