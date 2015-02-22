# embedding_zero-shot-learning

This is the code of %  Paper references:  Fu et al. Transductive Multi-view Embedding for Zero-Shot Recognition and Annotation, (ECCV 2014)

The codes include all the work in this paper, except:
the SVM projections: from low-level feature --> attribute;
since, this part is standard one and the low-level feature kernel matrix is very huge. One can do it by themselve.

I do include the feature matrix of overfeat and decaf in the mat file. And this case, we can use liblinear to predict word2vec or attribute vectors. One example is in deepLearning_example.m

word2vec is just one case of semantic word vectors. And we use wikpedia articles to train them. One may use other linguistic articles to train the models. The semantic word vectors would be different, ( I expect this). For example, google documents and etc al.  That means our framework is extendable on this point.

Another extendable point is that we can use other types of features to enable the whole pipeline. Please refer to my recent publication: Transductive Multi-view Zero-Shot Learning, accepted to TPAMI.

Anyway, I believe this framework would be more effecient and accurate to solve zero-shot learning problems in general than the other methods.



--Yanwei Fu, Feb 4th, 2015 
