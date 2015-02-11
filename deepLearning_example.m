load('./mat/deep_feat/AwA_overfeat.mat');
load('./mat/AwA100_word2vec.mat');
load('./mat/AwA_input.mat');
%%
wordattr = AwA100;  %AwA85word;

% generate word-training and word-testing attribute matrix:
te_cls_attr = wordattr(logical(animal_cls.testing_cls_flag),:);
tr_cls_attr = wordattr(logical(~animal_cls.testing_cls_flag),:);
    
te_inst_attr = wordattr(test_img_label,:);
tr_inst_attr = wordattr(train_img_label,:);

%%
for id = 1:size(AwA100,2)
    
    cmd =[sprintf(' -s 11  -c  10  ')];
    model =liblineartrain(tr_inst_attr(:,id), sparse(tr_overfeat), cmd);
    zslmodel100{id} =model;
    
    [plxte]=liblinearpredict(te_inst_attr(:,id),sparse(te_overfeat),model);
%     [plxtr]=liblinearpredict(tr_inst_attr(:,id),AwAfeat,model);
    zslpL_Xte(:,id) = plxte;
%     zslpL_Xtr(:,id) = plxtr;
end
%% try simply Nearest neighbour:
con_test_prototype = wordattr(logical(animal_cls.testing_cls_flag),:);

opts.top=500;
[conzsl_nrm_res,conzsl_euc_res]=Fu_ZSL_self_training_byNN(zslpL_Xte, con_test_prototype, test_img_label,zsl_label, opts);

