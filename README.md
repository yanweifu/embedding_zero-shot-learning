# embedding_zero-shot-learning

This is the code of %  Paper references:  Fu et al. Transductive Multi-view Embedding for Zero-Shot Recognition and Annotation, (ECCV 2014)

The codes include all the work in this paper, except:
the SVM projections: from low-level feature --> attribute;
since, this part is standard one and the low-level feature kernel matrix is very huge. One can do it by themselve.

I do include the feature matrix of overfeat and decaf in the mat file. And this case, we can use liblinear to predict word2vec or attribute vectors. One example is in deepLearning_example.m

--Yanwei Fu, Feb 4th, 2015 
