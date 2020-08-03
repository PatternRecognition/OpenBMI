function [data idx str] = qa_get(dat,str)
%
% USAGE:       function [data idx str] = qa_get(dat,str)
%
% IN:       dat         -       a epo or marker structure
%           str         -       a string or cell array of strings 
%                               specifying which events should be kept:
%                               'TP': true positives
%                               'FP': false positives
%                               'FN': false negatives
%                               'TN': true negatives
%
% OUT:      data        -       updated epo or marker structure
%           idx         -       indices of kept events   
%           str         -       (posibily reordered) input variable 'str'
%
% Simon Scholler, June 2011
%


if isfield(dat,'x')   % check if input is epo or marker struct
    isepo = 1;
else
    isepo = 0;
end

if ~iscell(str)
    bl_idx = 1;  % Convention: Baseline Class is the first in the Structure!
    others_idx = setdiff(1:length(dat.className),bl_idx);
    switch str
        case 'TP'   % true positives
            idx = find(sum(dat.y(others_idx,:),1) & dat.detected);
        case 'FP'   % false positives
            idx = find(dat.y(bl_idx,:) & dat.detected);
        case 'FN'   % false negatives
            idx = find(sum(dat.y(others_idx,:),1) & ~dat.detected);
        case 'TN'   % true negatives
            idx = find(dat.y(bl_idx,:) & ~dat.detected);
        otherwise
            error('Input string unknown.')
    end
    warning off
    if isepo
        data = proc_selectEpochs(dat,idx);
    else
        data = mrk_selectEvents(dat,idx);
    end
    warning on
else
    % sort cell array such that baseline class is the first in the
    % output structure
    fp = strmatch('FP',str);        
    if ~isempty(fp)
        str = str([fp 1:fp-1 fp+1:end]);
    end
    tn = strmatch('TN',str);
    if ~isempty(tn)
        str = str([tn 1:tn-1 tn+1:end]);
    end
    % get indices
    [data idx] = qa_get(dat,str{1});
    for n = 2:length(str)
        [dat_tmp ix] = qa_get(dat,str{n});
        idx = [idx ix];
        if isepo
            data = proc_appendEpochs(data,dat_tmp);
        else
            data = mrk_mergeMarkers(data,dat_tmp);
        end
    end
end

        