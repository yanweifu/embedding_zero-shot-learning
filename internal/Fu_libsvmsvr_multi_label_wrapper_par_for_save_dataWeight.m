function [pL_Xtr, pL_Xte, acc_te, acc_tr, te_au, te_r2, te_r0,  curve, models] = Fu_libsvmsvr_multi_label_wrapper_par_for_save_dataWeight(mat_name, Xtrain,train_attr,Xtest, test_attr,opts)
%
% [pL_Ytrain, pL_Ytest, acc_te, acc_tr, te_au, te_r2, te_r0,  curve, models] = Fu_libsvmsvr_multi_label_wrapper(Ytrain,train_attr,Ytest, test_attr,opts)
%  train_attr : No_instance*No_attr(no_cls); each class is trained an individual SVM.
%
%[c,g,bestcv,acc,cmat,bestmodel] = Fu_libsvmsvr_cv_kernel_wrapper(Xtrain,train_label,opts)
%
% opts = getPrmDflt(opts,{'kernel','chisq','norm_type','row','learn_type','svm'},-1);
%  opts: kernel --'chisq', 'intersect','linear';
%           norm_type: row, col, other
%           learn_type: svm, svr;
%
opts = getPrmDflt(opts,{'kernel','chisq','norm_type','row','learn_type','svm', ...
                                        'prob',true,'useProb', true,'input','bits', 'doEval',false, 'useWeight',true,'pre_trained_kernel',false},-1);


iif =@(varargin)varargin{2*find([varargin{1:2:end}],1,'first')}();
% data normalization functions:
ftrain =@(opt, Xtrain) iif(strcmp(opt.norm_type,'row'), CCV_normalize(Xtrain,1), ...
                          strcmp(opt.norm_type,'col'),  CCV_normalize(Xtrain,2), ...  % 'col'
                          1, Xtrain); % else no normalize

if ~opts.pre_trained_kernel
    % train the kernel.
    if strcmp(opts.kernel,'linear') ||strcmp(opts.kernel,'dotprod')
        [Ktrain,Ktest]= Fu_compute_kernel_matrices(ftrain(opts,Xtrain), ftrain(opts,Xtest),'dotprod');
    elseif strcmp(opts.kernel,'chisq')
        [Ktrain,Ktest]= Fu_compute_kernel_matrices(ftrain(opts,Xtrain), ftrain(opts,Xtest),'chisq');
    elseif strcmp(opts.kernel,'intersect')
        [Ktrain,Ktest]= Fu_compute_kernel_matrices(ftrain(opts,Xtrain), ftrain(opts, Xtest),'intersect');    
    end
else
    Ktrain = opts.Ktrain;
    Ktest= opts.Ktest;    
end
    
    
No_tr_inst = size(train_attr,1);
opts.K = [(1:No_tr_inst)'  Ktrain];
Ktrain = opts.K;
Ktest = [(1:size(Ktest,1))' Ktest];


opts.makeKernelPtr = @Fu_compute_kernel_matrices;
opts.customKernel = true;


if isfield(opts,'fixed_C')% need to cross-validate C;
    fprintf('Using pre-defined C: %f \n', opts.fixed_C);
    opts.C = opts.fixed_C;
else
    fprintf('Cross validate parameters C for SVM/SVR \n');
end

 
if strcmp(opts.learn_type,'svm')
    fprintf('use SVM learn-type, and use binary attribute input for each single attribute; \n');
    [pL_Xtr, pL_Xte,acc_te, acc_tr, te_au, te_r2, te_r0,  curve, models] = Fu_libsvm_fixed_multiattrib5Weight(Ktrain,train_attr,Ktest,test_attr,opts);
elseif strcmp(opts.learn_type,'svr')
    fprintf('use SVR learn-type and for each attribute SVR, use original value; \n');
    opts.input = 'continuous';
    [pL_Xtr, pL_Xte,acc_te, acc_tr, te_au, te_r2, te_r0,  curve, models] = Fu_libsvr_fixed_multiattrib5Weight(Ktrain,train_attr,Ktest,test_attr,opts);
end


save(mat_name,'pL_Xtr','pL_Xte','models','acc_tr','acc_te');


    

    

function [pL_Ytrain, pL_Ytest, acc_te, acc_tr, te_au, te_r2, te_r0,  curve, models] = Fu_libsvm_fixed_multiattrib5Weight(Ytrain,train_attr,Ytest,test_attr,opts)
% Train and optionally evaluate the specified multilabel dataset using independent 1-v-all SVM classifiers.
% function [pL_Ytrain, pL_Ytest, te_au, te_r2, te_r0,  curve, models] = libsvm_fixed_multiattrib4(Ytrain,train_attr,Ytest,test_attr,opts)
% Inputs:
%   Ytrain, train_attr: Train data must be same length each.
%   Ytest, test_attr: Test data.  (todo make optional).
%   opts.
%       kernel:     'chi-sq','RBF','linear' (todo histogram intersect)
%       prob:       binary. Probability estimates or not. Default 0.
%       doEval:     binary. Do evaluation or not.
%       parTrain:	Paralell train (requires open matlabpool)
%       useWeight:	Use weights to compensate for dataset imbalance
%       input:      'list','binary'. Each train instance is list of labels or Nc length bitvector.
%       fixedRetr:	(If doEval) how many fixed retreivals to assume. deflault = 2.
% Outputs:
%       pL_Ytrain:     Ntr x Nclass independent multiattrib posterior p(Attrib_i=true|x)
%       pL_Ytest:     Nte x Nclass independent multiattrib posterior p(Attrib_i=true|x)
%       te_au:      Area under label - detection curve for various statistics.
%       te_r2:      Rank N results (N = opts.fixedRetr)
%       te_r0:      Fixed threshold=0.5 results.
%       models:     Trained SVM model cell array (one per attrib) use for separate testing.
%
% Modified based on the Codes of Tim's libsvm_fixed_multiattrib4.m
%
%   libsvm_fixed_multiattrib4(Ytrain,train_attr,Ytest,test_attr,opts);
%
 te_au=0; te_r2=0; te_r0=0;  curve=0;
    if(nargin==4), 
        opts = struct(); 
    end
    opts = getPrmDflt(opts, {'kernel', 'RBF','prob',0,'fixedRetr',2,'input','bits','useWeight',1,'parTrain',0,'doEval',1}, -1);
    
    if(strcmp(opts.input,'list'))
        Nc = numel(setdiff(unique(train_attr),0));
    else
        Nc = size(train_attr,2);
        assert(size(train_attr,2)==size(test_attr,2));
    end
    Ntr = size(train_attr,1);
    Nte = size(test_attr,1);
    models = cell(1,Nc);
    
    pL_Ytrain = zeros(Ntr,Nc); 
    pL_Ytest = zeros(Nte,Nc);
    
    acc_te = 0;
    acc_tr = 0;
    if(opts.parTrain)
        parfor i = 1 : Nc    
            [pL_Ytrain(:,i), pL_Ytest(:,i), models{i},acc_te(i),acc_tr(i)] = learnClassWeight(i, Ytrain, train_attr, Ytest, test_attr, opts);
        end
    else
        for i = 1 : Nc    
            [pL_Ytrain(:,i), pL_Ytest(:,i), models{i}, acc_te(i),acc_tr(i)] = learnClassWeight(i, Ytrain, train_attr, Ytest, test_attr, opts);
        end
    end
    
    if(opts.doEval)
        [acc_r2, pr_r2, re_r2, fm_r2, ~, acx_r2] = util_evalLabeling(test_attr, pL_Ytest,  opts.fixedRetr);
        [acc_r0, pr_r0, re_r0, fm_r0, ~, acx_r0] = util_evalLabeling(test_attr, pL_Ytest,  0);
        [acc_au, pr_au, re_au, fm_au, curve, acx_au] = util_evalLabeling(test_attr, pL_Ytest, -1);

        te_r2 = [acc_r2, pr_r2, re_r2, fm_r2, acx_r2];
        te_r0 = [acc_r0, pr_r0, re_r0, fm_r0, acx_r0];
        te_au = [acc_au, pr_au, re_au, fm_au, acx_au];
    end
 function [pL_Ytrain, pL_Ytest, acc_te, acc_tr, te_au, te_r2, te_r0,  curve, models] = Fu_libsvr_fixed_multiattrib5Weight(Ytrain,train_attr,Ytest,test_attr,opts)
% Train and optionally evaluate the specified multilabel dataset using independent 1-v-all SVM classifiers.
% function [pL_Ytrain, pL_Ytest, te_au, te_r2, te_r0,  curve, models] = libsvm_fixed_multiattrib4(Ytrain,train_attr,Ytest,test_attr,opts)
% Inputs:
%   Ytrain, train_attr: Train data must be same length each.
%   Ytest, test_attr: Test data.  (todo make optional).
%   opts.
%       kernel:     'RBF','linear' 'chi-square' (todo histogram intersect)
%       prob:       binary. Probability estimates or not. Default 0.
%       doEval:     binary. Do evaluation or not.
%       parTrain:	Paralell train (requires open matlabpool)
%       useWeight:	Use weights to compensate for dataset imbalance
%       input:      'list','binary'. Each train instance is list of labels or Nc length bitvector.
%       fixedRetr:	(If doEval) how many fixed retreivals to assume. deflault = 2.
% Outputs:
%       pL_Ytrain:     Ntr x Nclass independent multiattrib posterior p(Attrib_i=true|x)
%       pL_Ytest:     Nte x Nclass independent multiattrib posterior p(Attrib_i=true|x)
%       te_au:      Area under label - detection curve for various statistics.
%       te_r2:      Rank N results (N = opts.fixedRetr)
%       te_r0:      Fixed threshold=0.5 results.
%       models:     Trained SVM model cell array (one per attrib) use for separate testing.

% Modified based on the Codes of Tim's libsvm_fixed_multiattrib4.m
%
%   libsvm_fixed_multiattrib4(Ytrain,train_attr,Ytest,test_attr,opts);
%
%  this is used for SVR method.
 te_au=0; te_r2=0; te_r0=0;  curve=0;
    if(nargin==4), 
        opts = struct(); 
    end
    opts = getPrmDflt(opts, {'kernel', 'RBF','prob',0,'fixedRetr',2,'input','bits','useWeight',1,'parTrain',0,'doEval',1,'svmtype',3}, -1);
    
    if(strcmp(opts.input,'list'))
        Nc = numel(setdiff(unique(train_attr),0));
    else
        Nc = size(train_attr,2);
        assert(size(train_attr,2)==size(test_attr,2));
    end
    Ntr = size(train_attr,1);
    Nte = size(test_attr,1);
    models = cell(1,Nc);
    
    pL_Ytrain = zeros(Ntr,Nc); 
    pL_Ytest = zeros(Nte,Nc);
    
    acc_te = 0;
    acc_tr = 0;
    if(opts.parTrain)
        parfor i = 1 : Nc    
            [pL_Ytrain(:,i), pL_Ytest(:,i), models{i},acc_te(i),acc_tr(i)] = learnClass_SVRWeight(i, Ytrain, train_attr, Ytest, test_attr, opts);
        end
    else
        for i = 1 : Nc    
            [pL_Ytrain(:,i), pL_Ytest(:,i), models{i}, acc_te(i),acc_tr(i)] = learnClass_SVRWeight(i, Ytrain, train_attr, Ytest, test_attr, opts);
        end
    end
    
    if(opts.doEval)
        [acc_r2, pr_r2, re_r2, fm_r2, ~, acx_r2] = util_evalLabeling(test_attr, pL_Ytest,  opts.fixedRetr);
        [acc_r0, pr_r0, re_r0, fm_r0, ~, acx_r0] = util_evalLabeling(test_attr, pL_Ytest,  0);
        [acc_au, pr_au, re_au, fm_au, curve, acx_au] = util_evalLabeling(test_attr, pL_Ytest, -1);

        te_r2 = [acc_r2, pr_r2, re_r2, fm_r2, acx_r2];
        te_r0 = [acc_r0, pr_r0, re_r0, fm_r0, acx_r0];
        te_au = [acc_au, pr_au, re_au, fm_au, acx_au];
    end
      
function [pL_Ytrain, pL_Ytest, model,acc_i_te,acc_i_tr] = learnClassWeight(i, Ytrain, train_attr, Ytest, test_attr, opts)
% Learning one binary SVM for each attribute
global c_value;

    Ntr = size(train_attr,1);   % number of training video instances
    Nte = size(test_attr,1);    % number of testing video instances
    if(strcmp(opts.input,'list'))
        pidx = sum(train_attr==i,2)>0;
        nidx = sum(train_attr==i,2)==0;
        Ytri = zeros(Ntr,1);
        Ytri(pidx) = 1;
        Ytri(nidx) = 0;        
        h=hist(Ytri,0:1);

        pidx = sum(test_attr==i,2)>0;
        nidx = sum(test_attr==i,2)==0;
        Ytei = zeros(Nte,1);
        Ytei(pidx) = 1;
        Ytei(nidx) = 0;
    elseif(strcmp(opts.input,'bits'))
        pidx = train_attr(:,i)==1;  % postive index
        nidx = train_attr(:,i)==0;  % negative index
        Ytri = zeros(Ntr,1);  
        Ytri(pidx) = 1;
        Ytri(nidx) = 0;        
        h=hist(Ytri,0:1);

        pidx = test_attr(:,i)==1;
        nidx = test_attr(:,i)==0;
        Ytei = zeros(Nte,1);
        Ytei(pidx) = 1;
        Ytei(nidx) = 0;
    elseif (strcmp(opts.input,'continuous'))
        Ytri = train_attr(:,i);
        Yte = test_attr(:,i);
        
    end

  %  w_c = 1./(h./max(h));
  %  weightstr = sprintf('-w%d %0.2f ', [0:1; w_c] );
  if ~strcmp(opts.input,'continuous') && opts.useWeight
     l_tr = length(Ytri);
     w_c = l_tr./(2.*h);
     weightstr = sprintf('-w%d %0.2f ', [0:1; w_c] );
  else      
      weightstr=' ';      
  end
  
    %Linear SVM
    if(strcmp(opts.kernel,'linear'));
        model = libsvmtrain(Ytri, Ytrain, '-t 0'); %No prob.          
        [~,~,vte] = svmpredict(Ytei, Ytest, model);
        [~,~,vtr] = svmpredict(Ytri, Ytrain, model);

        vals_tr(:,i) = vtr(:,1); % p(Not present).
        vals_te(:,i) = vte(:,1);
    end

    %RBF SVM with hyperparam opt.      
    if(strcmp(opts.kernel,'RBF'))
        opts.repeatFinalEst = 0;
        if(~opts.useWeight)
            weightstr = '';
        end
        %opts.cvMethod = 'explicit';
        opts.cvMethod = 'internal';
        [c,g] = Fu_libsvm_cv(Ytrain, Ytri, opts);
    end
    
    % specially for chi-square I use fixed C, if opts.C is provided. 
    if (strcmp(opts.kernel,'chi-square')||strcmp(opts.kernel,'chisq'))
        
        opts.repeatFinalEst = 0;
        if(~opts.useWeight)
            weightstr = '';
        end
        %opts.cvMethod = 'explicit';
        if (~isfield(opts,'C'))
            opts.cvMethod = 'internal';
            [c,g] = Fu_libsvm_cv(Ytrain, Ytri, opts);
        end
    end
        %No CV
        %model = libsvmtrain(Ytri, Ytrain, '-c 64');

        %Prob ests
        if(opts.prob)
            if strcmp(opts.kernel,'chi-square')||strcmp(opts.kernel,'chisq')
                if isfield(opts,'C')
                    model = libsvmtrainWeight(opts.beta,Ytri, Ytrain, sprintf('%s -c %1.2f  -t 4 -b 1', weightstr, opts.C));
                    c_value(i)=opts.C;
                else
                    model = libsvmtrainWeight(opts.beta, Ytri, Ytrain, sprintf('%s -c %1.2f  -t 4 -b 1', weightstr, c));
                    c_value(i) = c;
                end
            elseif strcmp(opts.kernel,'RBF')                
                model = libsvmtrain(Ytri, Ytrain, sprintf('%s -c %1.2f -g %1.2f -b 1', weightstr, c, g));
                c_value(i)=c;
            else
                model = libsvmtrain(Ytri, Ytrain, sprintf('%s -b 1', weightstr));
            end
                [~,acc_te,vte] = svmpredictWeight(Ytei, Ytest, model, '-b 1'); % p(Not present)> 
                [~,acc_tr,vtr] = svmpredictWeight(Ytri, Ytrain, model, '-b 1');
                vals_tr(:,i) = vtr(:,1) * (model.Label(1)==0) + (1-vtr(:,1)) * (model.Label(1)==1);
                vals_te(:,i) = vte(:,1) * (model.Label(1)==0) + (1-vte(:,1)) * (model.Label(1)==1);
                pL_Ytrain = 1-vals_tr(:,i);
                pL_Ytest = 1-vals_te(:,i);
        else
            % No ests.
            if strcmp(opts.kernel,'chi-square') 
                if isfield(opts,'C')
                    model = libsvmtrainWeight(opts.beta,Ytri, Ytrain, sprintf('%s -c %1.2f  -b 0 -t 4', weightstr, opts.C));
                    c_value(i) = opts.C;
                else
                    model = libsvmtrainWeight(opts.beta,Ytri, Ytrain, sprintf('%s -c %1.2f  -b 0 -t 4', weightstr, c));
                    c_value(i) = c;
                end
            elseif strcmp(opts.kernel,'RBF')                
                model = libsvmtrain(Ytri, Ytrain, sprintf('%s -c %1.2f -g %1.2f -b 0', weightstr, c, g));
            else
                model = libsvmtrain(Ytri, Ytrain, sprintf('%s -b 0', weightstr));
                
            end
%            model = libsvmtrain(Ytri, Ytrain, sprintf('%s -c %1.2f -g %1.2f -b 0', weightstr, c, g));
            [~,acc_te,vte] = svmpredictWeight(Ytei, Ytest, model, '-b 0');
            [~,acc_tr,vtr] = svmpredictWeight(Ytri, Ytrain, model, '-b 0');
            vals_tr(:,i) = vtr(:,1) * ( (model.Label(1)==0)*1 + (model.Label(1)==1)*-1) ; %Switch according to label. Sigh.
            vals_te(:,i) = vte(:,1) * ( (model.Label(1)==0)*1 + (model.Label(1)==1)*-1) ;
            pL_Ytrain = -vals_tr(:,i);
            pL_Ytest = -vals_te(:,i);
        end
    
        % I also want to save the accuracy of each attribute.
        acc_i_te =acc_te(1);
        acc_i_tr = acc_tr(1);
        
        
        
function [pL_Ytrain, pL_Ytest, model,acc_i_te,acc_i_tr] = learnClass_SVRWeight(i, Ytrain, train_attr, Ytest, test_attr, opts)
% Learning one binary SVM for each attribute
global c_value;

Ntr = size(train_attr,1);   % number of training video instances
Nte = size(test_attr,1);    % number of testing video instances
if(strcmp(opts.input,'list'))
pidx = sum(train_attr==i,2)>0;
nidx = sum(train_attr==i,2)==0;
Ytri = zeros(Ntr,1);
Ytri(pidx) = 1;
Ytri(nidx) = 0;        
h=hist(Ytri,0:1);

pidx = sum(test_attr==i,2)>0;
nidx = sum(test_attr==i,2)==0;
Ytei = zeros(Nte,1);
Ytei(pidx) = 1;
Ytei(nidx) = 0;
elseif(strcmp(opts.input,'bits'))
    pidx = train_attr(:,i)==1;  % postive index
    nidx = train_attr(:,i)==0;  % negative index
    Ytri = zeros(Ntr,1);  
    Ytri(pidx) = 1;
    Ytri(nidx) = 0;        
    h=hist(Ytri,0:1);

    pidx = test_attr(:,i)==1;
    nidx = test_attr(:,i)==0;
    Ytei = zeros(Nte,1);
    Ytei(pidx) = 1;
    Ytei(nidx) = 0;
elseif (strcmp(opts.input,'continuous'))
    Ytri = train_attr(:,i);
    Ytei = test_attr(:,i);

end

if ~strcmp(opts.input,'continuous') && opts.useWeight
 l_tr = length(Ytri);
 w_c = l_tr./(2.*h);
 weightstr = sprintf('-w%d %0.2f ', [0:1; w_c] );
else      
  weightstr=' ';      
end
  
%Linear SVM
if(strcmp(opts.kernel,'linear'));
model = libsvmtrain(Ytri, Ytrain, '-t 0'); %No prob.          
[~,~,vte] = svmpredict(Ytei, Ytest, model);
[~,~,vtr] = svmpredict(Ytri, Ytrain, model);

vals_tr(:,i) = vtr(:,1); % p(Not present).
vals_te(:,i) = vte(:,1);
end

%RBF SVM with hyperparam opt.      
if(strcmp(opts.kernel,'RBF'))
opts.repeatFinalEst = 0;
if(~opts.useWeight)
weightstr = '';
end
%opts.cvMethod = 'explicit';
opts.cvMethod = 'internal';
[c,g] = Fu_libsvr_cvWeight(Ytrain, Ytri, opts);
end
if (strcmp(opts.kernel,'chi-square')||strcmp(opts.kernel,'chisq'))

opts.repeatFinalEst = 0;
if(~opts.useWeight)
weightstr = '';
end
%opts.cvMethod = 'explicit';
    if (~isfield(opts,'C'))
        opts.cvMethod = 'internal';
        [c,g] = Fu_libsvr_cvWeight(Ytrain, Ytri, opts);
    end
end
%No CV
%model = libsvmtrain(Ytri, Ytrain, '-c 64');

%Prob ests
if(opts.prob)
    if strcmp(opts.kernel,'chi-square')||strcmp(opts.kernel,'chisq')
        %  KYtrain =[(1:size(Ytrain,1))', Ytrain];
        if isfield(opts,'C')
            model = libsvmtrainWeight(opts.beta,Ytri, Ytrain, sprintf('%s -c %1.2f  -t 4 -b 1 -s %s ', weightstr, opts.C,num2str(opts.svmtype)));
            c_value(i) = opts.C;
        else
            model = libsvmtrainWeight(opts.beta, Ytri, Ytrain, sprintf('%s -c %1.2f  -t 4 -b 1 -s %s ', weightstr, c,num2str(opts.svmtype)));
            c_value(i) = c;
        end
    elseif strcmp(opts.kernel,'RBF')     
              if isfield(opts,'C')   
                model = libsvmtrain(Ytri, Ytrain, sprintf('%s -c %1.2f -b 1 -s %s  ', weightstr, opts.C,  num2str(opts.svmtype)));
                c_value(i) = opts.C;
              else
                model = libsvmtrain(Ytri, Ytrain, sprintf('%s -c %1.2f  -b 1 -s %s  ', weightstr, c,  num2str(opts.svmtype)));
                c_value(i) = c;
              end      

    else
             model = libsvmtrain(Ytri, Ytrain, sprintf('%s -b 1 -s %s ', weightstr, num2str(opts.svmtype)));
    end

    [~,acc_te,vte] = svmpredict(double(Ytei), double(Ytest), model); % p(Not present)> 
    [~,acc_tr,vtr] = svmpredict(double(Ytri), double(Ytrain), model);
%        vals_tr(:,i) = vtr(:,1) * (model.Label(1)==0) + (1-vtr(:,1)) * (model.Label(1)==1);
%       vals_te(:,i) = vte(:,1) * (model.Label(1)==0) + (1-vte(:,1)) * (model.Label(1)==1);
%       pL_Ytrain = 1-vals_tr(:,i);
%       pL_Ytest = 1-vals_te(:,i);
    pL_Ytrain = vtr(:); pL_Ytest = vte(:);
else
% No ests.
if  strcmp(opts.kernel,'chi-square')||strcmp(opts.kernel,'chisq')
    %KYtrain =[(1:size(Ytrain,1))', Ytrain];
    if isfiled(opts,'C')
        model = libsvmtrainWeight(opts.beta,Ytri, Ytrain, sprintf('%s -c %1.2f  -b 0 -t 4 -s %s ', weightstr, opts.C, num2str(opts.svmtype)));
        c_value(i)=opts.C;
    else
        model = libsvmtrainWeight(opts.beta,Ytri, Ytrain, sprintf('%s -c %1.2f  -b 0 -t 4 -s %s ', weightstr, c, num2str(opts.svmtype)));
        c_value(i) = c;
    end
    
elseif strcmp(opts.kernel,'RBF')                
    if isfield(opts,'C')
         model = libsvmtrain(Ytri, Ytrain, sprintf('%s -c %1.2f  -b 0 -s %s ', weightstr, opts.C, num2str(opts.svmtype)));
         c_value(i) = optsC;
    else
           model = libsvmtrain(Ytri, Ytrain, sprintf('%s -c %1.2f  -b 0 -s %s ', weightstr, c,  num2str(opts.svmtype)));
           c_value(i) = c;
    end
else
         model = libsvmtrain(Ytri, Ytrain, sprintf('%s -b 0', weightstr));

end
%            model = libsvmtrain(Ytri, Ytrain, sprintf('%s -c %1.2f -g %1.2f -b 0', weightstr, c, g));
[~,acc_te,vte] = svmpredictWeight(double(Ytei), double(Ytest), model);
[~,acc_tr,vtr] = svmpredictWeight(double(Ytri), double(Ytrain), model);
%             vals_tr(:,i) = vtr(:,1) * ( (model.Label(1)==0)*1 + (model.Label(1)==1)*-1) ; %Switch according to label. Sigh.
%             vals_te(:,i) = vte(:,1) * ( (model.Label(1)==0)*1 + (model.Label(1)==1)*-1) ;
%             pL_Ytrain = -vals_tr(:,i);
%             pL_Ytest = -vals_te(:,i);
 pL_Ytrain = vtr(:); pL_Ytest = vte(:);
end

% I also want to save the accuracy of each attribute.
%acc_i_te =acc_te(1);
%acc_i_tr = acc_tr(1);
acc_i_te = 0; acc_i_tr = 0;


function matrix =CCV_normalize(matrix,flag)
% input:
%       matrix: feature vectors for CCV
%       flag = 1,normalize each row
%       flag = 2,normalize each column,
%       flag = 3,normalize the matrix by the maximum and minimum
%       flag = 4,normalize the matrix by floating maximum (99.9%)
% output:
%       X: normalized feature vectors
%
%
% Note that: We must consider some video missing all
% the features.(e.g. some video does not have MFCC features)

% eps: smoothing factor.
eps=1e-30; 

if flag ==1
    % normalize each row:
    row_sum=sum(matrix,2);
    row_div_sum =repmat(row_sum,1,size(matrix,2));
    matrix = matrix./(row_div_sum+eps);    
end

if flag ==2
    % normalize each column:
    col_sum=sum(matrix,1);
    col_div_sum = repmat(col_sum,size(matrix,1),1);
    matrix = matrix./(col_div_sum+eps);
end

if flag ==3
    % normalize the whole matrix by the minimum and maximum number.
    Max_num = max(matrix(:));
    Min_num = min(matrix(:));
    if Max_num~=0;
        matrix = (matrix-Min_num)./(Max_num-Min_num);
    end
end

if flag ==4
    % float upbound(maximum value) of feature_matrix
    max_perc=0.999;
    bin_No = hist(matrix(:),[min(matrix(:)):max(matrix(:))]);
    threshold = sum(bin_No)*max_perc;
    sum_cnt = 0;
    for i=1:length(bin_No)
        if sum_cnt>=threshold
            Max_num_bin = i;
            fprintf('Maximum bin is %d\n', i);
            break;
        else
            sum_cnt = sum_cnt + bin_No(i);
        end
    end
    
    matrix = (matrix-min(matrix(:)))./(Max_num_bin - min(matrix(:)));
    
    bigger = matrix>=1;
    matrix = matrix -matrix.*bigger+bigger;
    
end

function [c,g,bestcv,acc,cmat,bestmodel] = Fu_libsvr_cvWeight(Ktrain,train_label,opts)
% This function is to make cross validation and find the best model
%   this function only used for RBF kernel or customer-defined kernel
% Input:
%  Ktrain: Train data. Ytrain:(video_No * feature_dimension)
%  train_label: Train labels. train_video_label(video_No*1)
%  opts.
%    nFold: Number of folds to use. (def 2)
%    useWeight: Use weights to normalise data frequency? (Prevent degenerate solutions for very unbalanced datasets). (def 0)
%    optCrit: 'cvAcc' : libsv cross validation accuracy. (cvAcc)
%    repeatFinalEst: 1+ For test accuracy estimation, how many times to repeat random folding to average over noise. 
%    cvMethod: 'internal' (libsvm),'explicit' (manual).
%    outProb: boolean. Use prob estimates for final output model?
%
% Return:
%  Optimal params.
% Display:
%  Also displays nfold accuracy after optimization and associated confusion matricies.
%
% Todo: Absolute or mean accuracy optimization/result.
% * Requires libsvm to be added in path.
% Modified and Added from Tim's SVM Code.
%
% Aug. 24 ------
% this file is changed from Fu_libsvm_cv.m;
% the aim is to cross-validate and get the best SVR model
%

if(nargin==2)
    opts = struct;
end
% by default, svmtype=3 (SVR)
opts = getPrmDflt(opts, {'nFold', 2, 'useWeight', false, 'optCrit', 'cvAcc', 'avgBestCv',false, ...
                         'cvSlack', true, 'cvGamma', true, 'customKernel', false, 'repeatFinalEst', 1, ...
                         'specSlack', 0, 'specGamma', 0,'cvMethod','internal','outProb',1, 'svmtype', 3},-1 );

if(size(Ktrain,1)~=size(train_label,1))
    error('Ktrain and train_label have mismatching numbers of elements');
end
if(size(Ktrain,1)==0 || size(train_label,1)==0)
    error('Ktrain or train_label missing elements');
end
if(sum(strcmp(opts.optCrit,{'meanErr', 'cvAcc'}))~=1)
    error('Optimization criteria must be "meanErr" or "cvAcc"');
end
    
bestcv = 0;
nClass = numel(unique(train_label));


% if(opts.useWeight)
%     if(any(train_label==0)) %the class labels are counting from 0.
%         h=hist(train_label,0:nClass-1); % histogram positive examples.
%         n_total = length(train_label);
%     
%         w_pos = n_total./(2.*h);
%         weightstr = sprintf(' -w%d %0.2f ',[0:nClass-1; w_pos]);
%     else    %the class labels are counting from 1.
%         h=hist(train_label,1:nClass); % histogram positive examples.
%         n_total = length(train_label);
%     
%         w_pos = n_total./(2.*h);
%         weightstr = sprintf(' -w%d %0.2f ',[1:nClass; w_pos]);
%     end
% else
%     weightstr = '';
% end
weightstr = '';


c=1;g=1;

if(opts.cvGamma)
    llog2g = -5:3;
else
    llog2g = 0;
end
if(opts.cvSlack)
    llog2c = 1:7;
else
    llog2c = 0;
end
if(opts.specSlack)
    llog2c = log2(opts.specSlack);
end
if(opts.specGamma)
    llog2g = log2(opts.specGamma);
end

cv_all = zeros(6,6);
aac_all = zeros(6,6);
i=1;
for log2c = llog2c
    j=1;
    for log2g = llog2g
        
        if(opts.customKernel)
            cmd = [weightstr ' -v ', num2str(opts.nFold), ' -c ', num2str(2^log2c),'  -s ', num2str(opts.svmtype)];
            cv_all(i,j) = libsvmtrainWeight(opts.beta,double(train_label), double(opts.K), [cmd, ' -t 4 ']);
        else
            if(strcmp(opts.cvMethod,'internal'))
                cmd = [weightstr, ' -q -v ', num2str(opts.nFold), ' -c ', num2str(2^log2c), ' -g ', num2str(2^log2g), '  -s ', num2str(opts.svmtype)];
                cv_all(i,j) = libsvmtrain(train_label, Ktrain, cmd);
            elseif(strcmp(opts.cvMethod,'explicit'))
                opts2 = opts;
                opts2.repeatFinalEst = 2;
                opts2.specSlack = 2^log2c;
                opts2.specGamma = 2^log2g;
                cv_all(i,j) = libsvm_cv_cvacc(Ktrain, train_label, opts2);
            end
        end
        if(strcmp(opts.optCrit,'meanErr'))% || EXTRA)
            cmd = [weightstr, ' -q  -c ', num2str(2^log2c), ' -g ', num2str(2^log2g), '  -s ', num2str(opts.svmtype)];
            model = libsvmtrain(train_label, Ktrain, cmd);
            [lab] = svmpredict(train_label,Ktrain,model);
            cmat = confusion_matrix(2,train_label,lab);
            %for c = 1 : nClass
            %    aac(c) = sum((lab==c)&(train_label==c))/sum(train_label==c);
            %end
            aac_all(i,j) = mean(diag(cmat));
        end
        if(strcmp(opts.optCrit,'meanErr'))
            if(aac_all(i,j)>bestcv)
                bestcv = aac_all(i,j);
                c = 2^log2c;
                g = 2^log2g;
            end
        elseif(strcmp(opts.optCrit,'cvAcc'))
            if(cv_all(i,j)>bestcv)
                bestcv = cv_all(i,j);
                c = 2^log2c;
                g = 2^log2g;
                %fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, c, g, bestcv);
            end
        else
            error('Unknown optimization criteria');
        end
        j=j+1;
    end
    i=i+1;
end

if(isfield(opts,'avgBestCv') && opts.avgBestCv)
    [i,j]=find(cv_all==max(cv_all(:)));
    ic = round(mean(i));
    ig = round(mean(j));
    [LG,LC] = meshgrid(-5:3,-1:5);
    c = 2^LC(ic,ig);
    g = 2^LG(ic,ig);
end
    

%% To estimate actual performance, do own crossvalidation. Assume 2 fold for now :-/.
acc=0; cmat = zeros(nClass, nClass);
for i = 1 : opts.repeatFinalEst
    l  = randperm(numel(train_label));
    N  = numel(train_label)-ceil(numel(train_label)/opts.nFold);
    X1 = Ktrain(l(1:N),:); X2 = Ktrain(l(N+1:end),:);
    Y1 = train_label(l(1:N));   Y2 = train_label(l(N+1:end));    

    if(opts.customKernel)
        cmd = [weightstr,' -c ' num2str(c), '  -s ', num2str(opts.svmtype)];
        [xtr,xte]=opts.makeKernelPtr(X1, X1,'train');
        K1tr = [(1:numel(Y1))', xtr];
        [xtr,xte]=opts.makeKernelPtr(X1, X2,'test');
        K1te = [(1:numel(Y2))', xte]; 
        [xtr,xte]=opts.makeKernelPtr(X2, X2,'train');
        K2tr = [(1:numel(Y2))', xtr];
        [xtr,xte]=opts.makeKernelPtr(X2, X1,'test');
        K2te = [(1:numel(Y1))', xte]; 
        model = libsvmtrain(double(Y1), double(K1tr), [cmd, ' -t 4 ']);
        [lab1, acc1p] = svmpredict(double(Y2), double(K1te), model); 
        model = libsvmtrain(double(Y2), double(K2tr), [cmd, ' -t 4 ']);
        [lab2, acc2p] = svmpredict(double(Y1), double(K2te), model); 
    else
        cmd = [weightstr,' -c ' num2str(c) ' -g ' num2str(g), '  -s ', num2str(opts.svmtype)];
        model = libsvmtrain(Y1, X1, cmd);
        [lab1, acc1p] = svmpredict(Y2, X2, model); 
        model = libsvmtrain(Y2, X2, cmd);
        [lab2, acc2p] = svmpredict(Y1, X1, model);
    end
    cmat1 = confusion_matrix(nClass, Y2, lab1);
    acc1  = mean(diag(cmat1));
    cmat2 = confusion_matrix(nClass, Y1, lab2);
    acc2 = mean(diag(cmat2));    
    acc  = acc + 0.5*sum(acc1(1)+acc2(1))/opts.repeatFinalEst;
    %fprintf(1,'2-fold accuracy: %f / %f = %f\n', acc1(1), acc2(1), acc);
    cmat = cmat + 0.5*(cmat1+cmat2)/opts.repeatFinalEst;
end

if(opts.repeatFinalEst>0)
    fprintf(1,'%d-fold accuracy: %f.\n',opts.nFold, acc*100);
end

if(nargout>5)
    if(opts.outProb)
        probstr = ' -b 1 ';
    else
        probstr = ' -b 0 ';
    end
    if(opts.customKernel)
        bestmodel = libsvmtrainWeight(opts.beta,double(train_label), double(opts.K), [weightstr,' -c ' num2str(c),  probstr, '-t 4', '  -s ', num2str(opts.svmtype)]);
    else
        bestmodel = libsvmtrainWeight(opts.beta,train_label, Ktrain, [weightstr,' -c ' num2str(c), ' -g ' ,num2str(g), probstr,  '  -s ', num2str(opts.svmtype)]);
    end
end

