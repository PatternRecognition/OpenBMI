function [out, score] = applyClassifier_hsan(fv, model, C, idx1, idx2, idx3)
%out= applyClassifier(fv, classy, C, <idx>)
% out_act.results = applyClassifier(fv_test_act, 'RLDAshrink', out_act.C);
% tResult = applyClassifier(fv_test_mi, 'wr_multiClass', out_mi.C);
%Arguments:
%  FV   - struct of feature vectors
%  C    - struct of trained classifier, out of trainClassifier
%  IDX  - array of indices (of features vectors) to which the classifier
%         is applied
%
%Returns:
%  OUT  - array of classifier outputs

fv= proc_flaten(fv);
applyFcn= getApplyFuncName(model);  % apply_separatingHyperplane

if exist('idx1','var'),
    [out, score] = feval(applyFcn,C,fv.x,idx1,idx2,idx3); %   here
%     out = feval(applyFcn,C,fv.x,idx1,idx2,idx3); %   here
% else
%   if isnumeric(fv.x),
%     out= feval(applyFcn, C, fv.x(:,idx));
%   else
%     fv = setTrainset(fv,idx);
%     out = feval(applyFcn,C,fv.x);
%   end
end



