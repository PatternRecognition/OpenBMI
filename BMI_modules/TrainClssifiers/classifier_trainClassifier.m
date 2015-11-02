function [ out ] = classifier_trainClassifier( fv, varargin )
%PROC_TRAIN_CLASSIFIER Summary of this function goes here
%   Detailed explanation goes here
% opt=opt_proplistToStruct_lower(varargin{:});
if ~varargin{end}
    varargin=varargin{1,1}; %cross-validation procedures
end;

switch lower(varargin{1})
    case 'lda' %only binary class
        out.cf_param=train_RLDAshrink(fv.x,fv.mrk.logical_y);
        out.classifier='LDA';
end

end

