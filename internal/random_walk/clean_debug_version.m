% addpath:
addpath('/import/geb-experiments/yf300/FeatureActivePerception/internal/');
addpath_folder('/import/geb-experiments/yf300/FeatureActivePerception/internal/');
addpath('/import/geb-datasets/yanwei/latest_wiki_data/internal/');
addpath_folder('/import/geb-datasets/yanwei/latest_wiki_data/internal/');
% AwA data prototype:
load('/homes/yf300/AwA_input.mat');
con =load('/import/geb-datasets/yanwei/latest_wiki_data/100mat/AwA100_res.mat');
%bin = load('/import/geb-experiments/yf300/ZSL_latent/code/mat/AwA_SVM_85UD.mat');
bin = load('/import/geb-datasets/yanwei/latest_wiki_data/bin_attr/AwA_85SVMbin.mat');

% load prototype:
load('/import/geb-datasets/yanwei/latest_wiki_data/100mat/AwA100.mat');

bin_test_prototype = animal_cls.cls_attribute_binary(logical(animal_cls.testing_cls_flag),:);
con_test_prototype = AwA100(logical(animal_cls.testing_cls_flag),:);
zsl_label =find(animal_cls.testing_cls_flag);

%
v1=zscore(Xtest);
v2=con.all_pL_Xte; v3=bin.all_pL_Xte;

addpath('/import/geb-experiments/yf300/multiviewCCA/Temp/multi_view_learning/CCA/release/');
[W,D]=CCA3(v1,v2,v3);
%
DP = power(D,4);
P= [v1,v2,v3]*W*D; index = [ones(size(v1,2),1);ones(size(v2,2),1)*2;ones(size(v3,2),1)*3];
Pv1= v1*W(index==1,:)*DP;
Pv2 =v2*W(index==2,:)*DP;
Pv3 =v3*W(index==3,:)*DP;

% % updated bin and con prototype:
% binPro = bin_test_prototype*W(index==3,:)*DP;
% conPro = con_test_prototype*W(index==2,:)*DP;

% original bin prototype:
opts.Top =[1:800]; 
[binzsl_nrm_res,binzsl_euc_res]=Fu_ZSL_self_training_byNN(v3, bin_test_prototype, test_img_label,zsl_label, opts);

opts.Top =[1:800]; 
[conzsl_nrm_res,conzsl_euc_res]=Fu_ZSL_self_training_byNN(v2, con_test_prototype, test_img_label,zsl_label, opts);


binPro2 =  binzsl_nrm_res.max_new_conf_prototype*W(index==3,:)*DP;
conPro2 = conzsl_nrm_res.max_new_conf_prototype*W(index==2,:)*DP;

opts.Top =[1:5:1500]; 
[Pconzsl_nrm_bin_q_low,Pconzsl_euc_bin_q_low]=Fu_ZSL_self_training_byNN(Pv1, binPro2, test_img_label,zsl_label, opts);
%%
save('proj_res.mat','Pconzsl_nrm_bin_q_low','conzsl_nrm_res','binzsl_nrm_res');

%%
% some settings for labels:
zsl_label = zsl_label(:);
test_img_label=test_img_label(:);

% prototype:
lowPro2 = Pconzsl_nrm_bin_q_low.max_new_conf_prototype;
binPro2; 
conPro2;
%  extended prototype:
ePv1 =[Pv1; lowPro2]; ePv2 =[Pv2; conPro2]; ePv3 = [Pv3;binPro2];
no = size(Pv1,1);

%  extended prototype:
No_test_cls = 10;

L = [zeros(no,No_test_cls);  eye(No_test_cls)]; 

k=10;
alpha = 0.25;
[sLabels, cLabels, sL1Labels, cL1Labels] = mvssl_v2(ePv1', ePv2', ePv3', L, k, alpha, no);


for i = 1:3
    zsLabels{i} =zsl_label(sLabels{i});
    zsL1Labels{i} = zsl_label(sL1Labels{i});    
end

fprintf('L2-SSL results:\n');
%show_result(sLabels, cLabels, test_label, no);
show_result(zsLabels, zsl_label(cLabels), test_img_label);

% acc_combined_conf =
% 
%     0.4073
% 
% ans =
% 
%     0.3188    0.3764    0.3851


fprintf('L1-SSL results:\n');
show_result(zsL1Labels, zsl_label(cL1Labels), test_img_label);