function [C,params]= trainClassifier_hsan(fv, classy, idx1, idx2, idx3 )
%C= trainClassifier(fv, classy, <idx>)

fv= proc_flaten(fv);
% if exist('idx', 'var'), 
%   if ~isnumeric(fv.x)
%     fv = setTrainset(fv,idx);
%   else 
%     fv.x= fv.x(:,idx);
%     fv.y= fv.y(:,idx);
%   end
% end

if isstruct(classy),
%% classifier is given as model with free model parameters
  model= classy;
%  classy= selectModel(fv, model);
classy= select_model(fv, model);
end

[func, params]= getFuncParam(classy);
trainFcn= ['train_' func];
if isfield(fv,'classifier_param')
    if exist('idx1','var')
        C= feval(trainFcn, fv.x, fv.y, idx1,idx2,idx3, fv.classifier_param{:}, params{:});   % train_wr_multiclass_hsan
    else
        C= feval(trainFcn, fv.x, fv.y, fv.classifier_param{:}, params{:});
    end
else
  C= feval(trainFcn, fv.x, fv.y, params{:});
end