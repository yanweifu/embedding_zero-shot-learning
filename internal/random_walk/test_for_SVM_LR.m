% testing by using SVR/LR

%% do cosine normalization:
sdata{1} = Pv1'; sdata{2} = Pv2'; sdata{3} = Pv3';
% do square sum-to-1 normalization:
for i=1:3
    X = sdata{i};
    XtX = X'*X;
    data = sqrt(diag(XtX));    
%    gK{i} = X./(X_l2norm{i}*X_l2norm{i}');
    sdata{i} = X./repmat(data', size(X,1),1);
end
    
clear proto
proto{1} = lowPro2'; proto{2}=conPro2'; proto{3} = binPro2';
for i =1:3
     X = proto{i};
    XtX = X'*X;
    data = sqrt(diag(XtX));    
%    gK{i} = X./(X_l2norm{i}*X_l2norm{i}');
    proto{i} = X./repmat(data', size(X,1),1);
end
%% 

addpath('/import/geb-datasets/yanwei/latest_wiki_data/multigraphZSL/internal/');

% opt.distance = 'sqdist'; opt.ntop =100;
% 
% newproto12 = doCluster(proto{1}, sdata{2}, opt);
% newproto13 = doCluster(proto{1}, sdata{3}, opt);
% newproto21 = doCluster(proto{2}, sdata{1}, opt);
% newproto23 = doCluster(proto{2}, sdata{3}, opt);
% newproto31 = doCluster(proto{3}, sdata{1}, opt);
% newproto32 = doCluster(proto{3}, sdata{2}, opt);
opt.distance = 'nrmcorr'; opt.ntop =100;
newproto12 = doCluster(proto{1}, Pv2', opt);
newproto13 = doCluster(proto{1}, Pv3', opt);
newproto21 = doCluster(proto{2}, Pv1', opt);
newproto23 = doCluster(proto{2}, Pv3', opt);
newproto31 = doCluster(proto{3}, Pv1', opt);
newproto32 = doCluster(proto{3}, Pv2', opt);

[cproto12] = doCosNrm(newproto12);
[cproto13] = doCosNrm(newproto13);
[cproto21] = doCosNrm(newproto21);
[cproto23] = doCosNrm(newproto23);
[cproto31] = doCosNrm(newproto31);
[cproto32] = doCosNrm(newproto32);

% train data:
train =[proto{1}, proto{2},proto{3},cproto12, cproto13, cproto21,cproto23,cproto31,cproto32];


train_label = repmat(zsl_label,9,1);


%% do linear svm:
addpath('/import/geb-experiments/yf300/FeatureActivePerception/internal/');
addpath_folder('/import/geb-experiments/yf300/FeatureActivePerception/internal/');

opt.norm_type = 'other';
opt.kernel ='linear';

[acc, prob_estimates_mfcc, bestmodel_mfcc, class_label_mfcc, meta_res_mfcc]=Fu_direct_SVM2(train', sdata{1}', train_label,test_img_label,opt);


%% LR:
opts.nFold = 3;
opts.specLam = 0;

strain = train(1:200,:);
ssdata{2} = sdata{1}(1:200,:);

[acctr,~,~,acc_te,cmat]=Fu_liblr_cv_tim(strain',train_label, opts, ssdata{2}',test_img_label);

%%
%top= [100,500,1000,2000,3500, 4500];

top =[10:30:200, 300:100:1000 1100:300:2000];
trainx=[ proto{1}, proto{2},proto{3}];
for i = 1:length(top)
opt.ntop =top(i);

opt.distance = 'nrmcorr'; 
newproto12 = doCluster(proto{1}, Pv2', opt);
newproto13 = doCluster(proto{1}, Pv3', opt);
newproto21 = doCluster(proto{2}, Pv1', opt);
newproto23 = doCluster(proto{2}, Pv3', opt);
newproto31 = doCluster(proto{3}, Pv1', opt);
newproto32 = doCluster(proto{3}, Pv2', opt);

[cproto12] = doCosNrm(newproto12);
[cproto13] = doCosNrm(newproto13);
[cproto21] = doCosNrm(newproto21);
[cproto23] = doCosNrm(newproto23);
[cproto31] = doCosNrm(newproto31);
[cproto32] = doCosNrm(newproto32);

% train data:
trainx =[trainx,cproto12, cproto13, cproto21,cproto23,cproto31,cproto32];

end

train_label = repmat(zsl_label,length(top)*6+3,1);

addpath('/import/geb-experiments/yf300/FeatureActivePerception/internal/');
addpath_folder('/import/geb-experiments/yf300/FeatureActivePerception/internal/');

opt.norm_type = 'other';


opt.kernel = 'linear';

opt.kernel = 'chisq';

[acc, prob_estimates_mfcc, bestmodel_mfcc, class_label_mfcc, meta_res_mfcc]=Fu_direct_SVM2(trainx(1:300,:)', sdata{2}(1:300,:)', train_label,test_img_label,opt);

%% absolute accuracy:
% 
[class_label, accuracy,prob_estimates] = svmpredict(test_img_label, ...
                                 sdata{1}(1:300,:)', bestmodel_mfcc, '-b 1');
%
[class_label, accuracy,prob_estimates] = svmpredict(test_img_label, ...
                                 sdata{3}(1:300,:)', bestmodel_mfcc, '-b 1');
