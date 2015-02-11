function [AUC,grdth,lab] = Fu_AUC(groundtruth,prob,opts)
%
%[AUC,grdth,lab] = Fu_AUC(groundtruth,prob,opts)
% Calculate the Area under curve
% Input: 
%       groundtruth: truth label
%       prob: the probabilities calculated, used for drawing ROC based on
%       different threshold.
%       the input should be a column vector.

% Output:
%       AUC: Area under curve

% If there are multiple classes, then we will calculate
% the AUC for each binary class.




if isfield(opts,'singleclass4video')&& opts.singleclass4video
    % change the values into the forms that can be invoked by plotroc,
% roc and plotconfusion functions.
grdth = Fu_plot_convert(groundtruth);

    % for multiple class problem, we only save the highest probability.
    [val,ind]=max(prob,[],2);
    lab = zeros(size(prob));
    col = size(prob,1);
    for i =1:col
        lab(i,ind(i))=val(i);
    end
    lab =lab';  % 每一列对应一个Instance。 此为roc的输入。
else
    lab = prob';
end
    

No_cls = max(groundtruth(:));
[TPR,FPR,TH]=roc(groundtruth,lab(:));

AUC = zeros(No_cls,1);


for i =1:No_cls
    if isfield(opts,'singleclass4video')&&opts.singleclass4video
        no = size(FPR{i},2);
        FPR{i}(no+1) = 1;
        TPR{i}(no+1) = 1;
    end    
    AUC(i) = trapz(FPR{i}', TPR{i}');
end

