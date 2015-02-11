function [zsl_nrmcorr_res,zsl_euc_res]=Fu_ZSL_self_training_byNN(pL_Xte, prototype,te_label,prototype_cls_label, opts)
%  [zsl_nrmcorr_res,zsl_euc_res]=ZSL_by_self_training_NN(pL_Xte, prototype,te_label,prototype_cls_label, opts)
%
% this is an improved and generalized version of ZSL_clsfier_by_NN.m
%
% if not specified the training label, just using the training label of the unique length of zsl_label
%

if nargin<5
    opts = getPrmDflt(opts,{'Top',100},-1);
elseif nargin<4
    tr_label =[1:length(unique(te_label))];
    tr_label=tr_label(:);
end

% number of testing label
ncls = numel(unique(te_label));
% maximum testing class label:
maxcls =max(te_label);

% do cosine similarity of prob_zsl (probability of testing zero-shot learning) in a self-training way:
distance='nrmcorr';
max_acc =0; max_prob=0;
for idx =1: length(opts.Top)
 ntop = opts.Top(idx);
% note that each row of prototype (class-prototype) should be corresponding to each class-label in tr_label.
cos_dis = slmetric_pw(prototype',pL_Xte',distance);
clear new_prot
% get new prototype by self-training:
for i= 1:size(prototype,1)
    [val,idx]=sort(cos_dis(i,:),'descend');
    new_prot(i,:)=mean(pL_Xte(idx(1:ntop),:),1);
end

% do classification using new prototype:
new_dis = slmetric_pw(new_prot',pL_Xte',distance);
[a,id]=max(new_dis);
%acc_nrmcorr(ntop)=sum(prototype_cls_label(id(:))==te_label)/length(te_label);

conf_1 =confusion_matrix(maxcls,te_label,prototype_cls_label(id(:)));
conf_1(isnan(conf_1))=0;
acc_confcorr(ntop) = sum(diag(conf_1))/ncls;

if acc_confcorr(ntop)>max_acc
    max_acc = acc_confcorr(ntop);
    % max_prob is the probability confidence of each instance belonging to each class.
    max_prob=new_dis;
    max_new_prototype= new_prot;
    end
end

[~,lb]=max(max_prob);
% predicted class label:
pred_cls_label = prototype_cls_label(lb);

zsl_nrmcorr_res.max_conf_prob = max_prob;
zsl_nrmcorr_res.pred_cls_label = pred_cls_label; % predicted class labels;
zsl_nrmcorr_res.max_conf_acc = max_acc;  % the maximum accuracy
%zsl_nrmcorr_res.acc_nrmcorr = acc_nrmcorr;
zsl_nrmcorr_res.max_new_conf_prototype = max_new_prototype;
zsl_nrmcorr_res.acc_confcorr =acc_confcorr;

conf_acc = confusion_matrix(maxcls,te_label,pred_cls_label);
conf_acc(isnan(conf_acc))=0;

zsl_nrmcorr_res.conf_acc = sum(diag(conf_acc))/ncls;
zsl_nrmcorr_res.conf_matrix = conf_acc;

% without using self-training way, directly comparing the prototype of each class for each instance;
new_dis = slmetric_pw(prototype',pL_Xte',distance);
[a,idorg]=max(new_dis);

%acc_org_nrmcorr(1)=sum(prototype_cls_label(id(:))==te_label)/length(te_label);

acc_org_nrmcorr_conf = confusion_matrix(maxcls,te_label,prototype_cls_label(idorg(:)));
acc_org_nrmcorr_conf(isnan(acc_org_nrmcorr_conf))=0;

zsl_nrmcorr_res.acc_org_conf =sum(diag(acc_org_nrmcorr_conf))/ncls;
zsl_nrmcorr_res.acc_org_conf_prob = new_dis;

% directly use the Euclidean distance:
norm_pL_Xte =CCV_normalize(pL_Xte,1); norm_test_prototype=CCV_normalize(prototype,1);
distance='eucdist';
max_euc_acc = 0; max_euc_prob =0; max_new_euc_proto =0;
for idx = 1:length(opts.Top);
    ntop = opts.Top(idx);
    
    cos_dis = slmetric_pw(norm_test_prototype',norm_pL_Xte',distance);
    clear new_prot
    
for i= 1:size(norm_test_prototype,1)
    [val,idx]=sort(cos_dis(i,:),'ascend');
    new_prot(i,:)=mean(norm_pL_Xte(idx(1:ntop),:),1);
end

new_dis = slmetric_pw(new_prot',norm_pL_Xte',distance);
[a,id]=min(new_dis);

conf_euc = confusion_matrix(maxcls,te_label,prototype_cls_label(id(:)));
conf_euc(isnan(conf_euc))=0;
acc_euc_conf(ntop)= sum(diag(conf_euc))/ncls;


%acc_euc(ntop)=sum(prototype_cls_label(id(:))==te_label)/length(te_label);

if acc_euc_conf(ntop)>max_euc_acc 
    max_euc_acc = acc_euc_conf(ntop);
    max_euc_prob = new_dis;
    max_euc_proto = new_prot;
end
end

% original center:
org_euc_dist = slmetric_pw(prototype',pL_Xte','eucdist');
[a,id]=min(org_euc_dist);
%acc_org_euc = sum(prototype_cls_label(id(:))==te_label)/length(te_label);
pred_euc_label = prototype_cls_label(id);
conf_euc_org = confusion_matrix(maxcls, te_label, pred_euc_label);
conf_euc_org(isnan(conf_euc_org)) =0;

[~,id]=max(max_euc_prob);
pred_cls_label = prototype_cls_label(id);
conf_euc_acc_final =confusion_matrix(maxcls, te_label,pred_cls_label(:));
conf_euc_acc_final(isnan(conf_euc_acc_final)) = 0;
zsl_euc_res.conf = conf_euc_acc_final;


zsl_euc_res.acc_euc = acc_euc_conf;
zsl_euc_res.max_new_prototype = max_euc_proto;
zsl_euc_res.max_acc = max_euc_acc;
zsl_euc_res.max_prob = max_euc_prob;
zsl_euc_res.acc_org_euc = sum(diag(conf_euc_org))/ncls; 


% debug.acc_nrmcorr =acc_nrmcorr;
% debug.acc_org_nrmcorr = acc_org_nrmcorr;
% debug.acc_euc = acc_euc;