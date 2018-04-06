
# Which optimizer and learning rate should I use for deep learning?

A common problem we all face when working on deep learning projects is chosing hyper-parameters. If you’re like me, you find yourself guessing an optimizer and learning rate, then checking if they work (and we’re not alone). This is laborious and error prone.

To better understand the affect of optimizer and learning rate choice, I trained the same model 500 times. The results show that the right hyper-parameters are crucial to training success.

In this article I’ll show the results of training the same model across 6 different optimizers and 48 different learning rates. I’ll also show the results of how scaling the model up 10x affects its training on fixed hyper-parameters.

The results show that:
- Most learning-rates will fail to train the model
- Training time vs learning rate exhibits a “valley” shape with the fastest training occuring in a narrow band of learning rates
- Each optimizer has a different optimal learning rate
- No one learning rate will successfully train across all optimizers tested

[Read the full article](https://medium.com/octavian-ai)
[See the code on GitHub](https://github.com/Octavian-ai/learning-rates)
[Run the code on FloydHub](https://www.floydhub.com/davidmack/projects/learning-rates/)
[Octavian.ai](https://octavian.ai)

