function [accnoprior, accwithprior,confnoprior, confwithprior,pAttrib,lpC] = Fu_DAP(tr_cls_attr, te_cls_attr, pL_Xte_85UD,test_img_label, zsl_label)
%
%  [accnoprior, accwithprior,lpC ] = Fu_DAP(tr_cls_attr, te_cls_attr, pL_Xte_85UD,test_img_label,zsl_label)
%
% DAP:
% groundtruth of training class-attribute matrix
%tr_cls_attr = logical(animal_cls.cls_attribute_binary(train_cls_flag,:));
%te_cls_attr = logical(animal_cls.cls_attribute_binary(logical(animal_cls.testing_cls_flag),:));
%
% pAttrib, is the log-prior for each attribute.
%
% lpC: the log prior of each class
%
[No_trcls,No_attr ]= size(tr_cls_attr);

pr_p_1 =zeros(1,No_attr);  % probability of empirical training data for attribute ==1;
pr_p_0 = zeros(1,No_attr); % probability of empirical training data for attribute ==0;

%Attrib prior.
pAttrib = sum(tr_cls_attr,1)/No_trcls; 

no_te_cls = length(zsl_label);

% prior_each_cls = pos_te_cls_attr*pr_p_1' + neg_te_cls_attr*pr_p_0';
lpC = zeros(1,no_te_cls);
for i=1:10     
     lpC(i) = sum(log(pAttrib(logical(te_cls_attr(i,:))))) + sum(log(1-pAttrib(~te_cls_attr(i,:))));
end
 
 %%  denum
  pos_te_pr = pL_Xte_85UD;  %probability of p(a|x)==1;
 neg_te_pr = 1-pL_Xte_85UD; % probability of p(a|x) ==0;
 
 no_te = size(pL_Xte_85UD,1);
 no_te_cls =length(unique(test_img_label));
 
 lpC_D = zeros(no_te,no_te_cls);
 lpC_D2= zeros(no_te,no_te_cls);
 
 % get the probability of each class, based on class ground truth
 for i =1:no_te_cls
      num = size(pos_te_pr,1);
      lpC_D(:,i) = sum(log(pos_te_pr(:,logical(te_cls_attr(i,:)))),2) + sum(log(neg_te_pr(:,~te_cls_attr(i,:))),2); %No prior.
      lpC_D2(:,i) = sum(log(pos_te_pr(:,logical(te_cls_attr(i,:)))),2) + sum(log(neg_te_pr(:,~te_cls_attr(i,:))),2) - lpC(i); %Prior.
 end
 mapping = unique(test_img_label);
 %%
     [~,idx] = max(lpC_D,[],2);
     
%      te_img_label= Fu_change_class_label(test_img_label,0);
     pred_img = mapping(idx);
     pred_img = pred_img(:);
     test_img_label = test_img_label(:);
    maxcls= max(test_img_label);
    confres = confusion_matrix(maxcls,test_img_label, pred_img);
    confres(isnan(confres))=0;
%    acc_euc_conf(ntop)= sum(diag(conf_euc))/ncls;
%     accnoprior = sum(pred_img == test_img_label);
     accnoprior = sum(diag(confres))/length(mapping);
      confnoprior = confres(mapping,mapping);
     
     [~,idx2] = max(lpC_D2,[],2);
     pred_img2 = mapping(idx2);
     pred_img2 = pred_img2(:);
     test_img_label = test_img_label(:);
    
      confres2 = confusion_matrix(maxcls,test_img_label,pred_img2);
      confres2(isnan(confres2))=0;
%      accwithprior = sum(pred_img2 == test_img_label);
     accwithprior = sum(diag(confres2))/length(mapping);
     
     confwithprior = confres2(mapping, mapping);