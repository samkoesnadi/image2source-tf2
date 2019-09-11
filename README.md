# image2source

Convert image to source code. In this project, it is specified to HTML+CSS.


## Editable configuration (common_definitions.py)
+ IS_TRAINING : Boolean

### Intermediate Fazit
11.09.2019: 21.23 :: 
+ A lot of variation of dataset is needed. Otherwise, the network will learn
pattern that is not what we intend it to learn. Hence, stuck in a loss with
no other way to learn (remember, there are sometimes 2 ways to achieve one thing,
but one way is better than the other). More data!
+ Fine-tuning the image feature extractor changes the feature completely from result
given by MobileNetV2 Autoencoder, and it changes the value range as well.


### TODO List
+ Window size is implemented, but not yet properly working. The only reason
window size is implemented is to reduce computation time. If it is not necessary,
then it can be left. The algorithm is to be improved.
