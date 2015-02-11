%  Paper references:
%[1] Fu et al. Learning Multi-modal Latent Attributes, TPAMI 2012
%[2] Fu et al. Transductive Multi-view Embedding for Zero-Shot Recognition and Annotation, (ECCV 2014)
%[3] Fu et al. Transductive multi-view zero-shot learning, submit to TPAMI 2015
%[4] Attribute-Based Classification for Zero-Shot Visual Object Categorization, TPAMI 2014

%% addpath:
addpath('./internal/');
addpath_folder('./internal/');

%% Load data:
% AwA data prototype:
load('./mat/AwA_input.mat');

% the SVR predictions from low-level features to 100-dim word2vec vectors 
con =load('./mat/AwA100_word2vec.mat');

% SVM predictions from low-level features to 85-dim attribute vectors:
bin =load('./mat/AwA_attr_res.mat','pL_Xte3');
bin.all_pL_Xte = bin.pL_Xte3;

% load prototype of 100-dim word2vec of all 50 AwA classes:
load('./mat/prototypes/AwA100.mat');
 
% generating the prototoypes of testing classes for both 100-dim word2vec
% and 85-dim attributes.
bin_test_prototype = animal_cls.cls_attribute_binary(logical(animal_cls.testing_cls_flag),:);
con_test_prototype = AwA100(logical(animal_cls.testing_cls_flag),:);
zsl_label =find(animal_cls.testing_cls_flag);

%% do the transductive embedding space:
v1=zscore(Xtest);
v2=con.all_pL_Xte;
v3=bin.all_pL_Xte;

[W,D]=CCA3(v1,v2,v3);
DP = power(D,4);
index = [ones(size(v1,2),1);ones(size(v2,2),1)*2;ones(size(v3,2),1)*3];
% add weighting to each dimension of embedding space -- [2,3]
Pv1= v1*W(index==1,:)*DP;
Pv2 =v2*W(index==2,:)*DP;
Pv3 =v3*W(index==3,:)*DP;

%% do 1-step self-training to update the prototypes in original spaces: 
% referring to [1]: Sec. semi-latent zero-shot learning
opts.Top =500; 
[binzsl_nrm_res,binzsl_euc_res]=Fu_ZSL_self_training_byNN(v3, bin_test_prototype, test_img_label,zsl_label, opts);
[conzsl_nrm_res,conzsl_euc_res]=Fu_ZSL_self_training_byNN(v2, con_test_prototype, test_img_label,zsl_label, opts);

binPro2 =  binzsl_nrm_res.max_new_conf_prototype*W(index==3,:)*DP;
conPro2 = conzsl_nrm_res.max_new_conf_prototype*W(index==2,:)*DP;

%% Try DAP directly [4]
% DAP results:
train_cls_flag = ~animal_cls.testing_cls_flag;
 tr_cls_attr = logical(animal_cls.cls_attribute_binary(train_cls_flag,:));
 te_cls_attr = logical(animal_cls.cls_attribute_binary(logical(animal_cls.testing_cls_flag),:));
[accnoprior3, accwithprior3,pAttrib, lpC]=Fu_DAP(tr_cls_attr, te_cls_attr,bin.all_pL_Xte,test_img_label,zsl_label);

%% let's do some random walk here:
% data preparation:
zsl_label = zsl_label(:);
test_img_label=test_img_label(:);

% prototype:
lowPro2 = (binPro2+conPro2)/2;%Pconzsl_nrm_con_q_low.max_new_conf_prototype;
binPro2; 
conPro2;
%  extended view matrix by prototypes:
ePv1 =[Pv1; lowPro2]; ePv2 =[Pv2; conPro2]; ePv3 = [Pv3;binPro2];
no = size(Pv1,1);

% total number of classes
maxcls = 50;

%  number of testing classes
No_test_cls = 10;

% do it with two views:
vdata{1} = ePv1';
vdata{2} = ePv2';
vdata{3} = ePv3';

gamma = 0.8; 
L = [zeros(no,No_test_cls);  2*eye(No_test_cls)-1]; 

opts =struct();
opts.k=30;

%[sLabels, cLabels, cZmat] = Fu_mvsslL2_v2( L,  vdata, opts);
opts.prototypeTime=8;
[cLabels,   Zmat, cW] = Fu_mvsslL2_v2_moreconnection_prototypes( L,  vdata, opts);


alpha =0.25; 
% combining them together:
L2 = L;
clear acc_sCombine
for itr = 1:10
    % clamping:
    %L2(6181:end,:) = [2*eye(No_test_cls)-1];
    
    L2= (1-alpha)*Zmat*L2;
    [~, sLabels] = max(L2(1:no,:), [], 2);
    Sres= confusion_matrix(maxcls, test_img_label(:),zsl_label(sLabels));    
    Sres(isnan(Sres)) = 0;
    acc_sCombine(itr)= sum(diag(Sres))/10;
end

%% N-shot example: 
%
% load N-shot data:
load('./mat/AwANshotidex15203050.mat','AwANshot');      

ins =[1:5,10 15 20 30 50 ];
rd = 10;

cZmat = Zmat;

dim = 100;
total = 1:6180;
for t = 1:length(ins)
    for r= 1:rd
    tridx=   AwANshot{ins(t)}.Ntr{r};
    teidx = setdiff(total, tridx);
    
    tr_label = test_img_label(tridx);
    te_label = test_img_label(teidx);
    tr_label = tr_label(:); te_label = te_label(:);
              
L = [zeros(no,No_test_cls);  2*eye(No_test_cls)-1]; 
for s = 1:length(zsl_label)
    L(tridx(:),:)=-1;
end
for s = 1:length(zsl_label)
    L(tridx(s,:),s)=1;
end
% combining them together:
L2 = L;
clear acc_sCombine
for itr = 1:20
   
    L2= (1-alpha)*cZmat*L2;
    [~, sLabels] = max(L2(teidx,:), [], 2);
    Sres= confusion_matrix(maxcls, te_label(:),zsl_label(sLabels));    
    Sres(isnan(Sres)) = 0;
    acc_sCombine(itr)= sum(diag(Sres))/10;
end    
 respv(ins(t),r) = max(acc_sCombine);   
    end
end

t=mean(respv,2);
t(t>0)


