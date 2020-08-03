function epo= proc_selectEpochs(epo, idx, exclude, varargin)
%epo= proc_selectEpochs(epo, idx)
%epo= proc_selectEpochs(epo, 'not', idx)
%
% selects the epochs with specified indices. void classes are removed.
%
% the structure 'epo' may contain a field 'indexedByEpochs' being a
% cell array of field names of epo. in this case subarrays of those
% fields are selected. here it is assumed that the last dimension
% is indexed by epochs.
%
% IN  epo  - structure of epoched data
%     idx  - indices of epochs that are to be selected [or are to be
%            excluded if the 'not' form is used].
%            if this argument is omitted, all non-rejected epochs are
%            selected, i.e., epochs with any(epo.y).
%
% OUT epo  - updated data structure

% bb 03/03, ida.first.fhg.de

if exist('exclude','var')
  if mod(length(varargin),2) == 0
    if isequal(idx, 'not'),
      idx= setdiff(1:size(epo.y,2), exclude);
    else
      error('if 3rd argument is given, the 2nd must be ''not''');
    end
  else   
    varargin = cat(2, exclude, varargin);    
  end
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'removevoidclasses', 1);

if ~exist('idx','var'),
  %% select accepted epochs
  idx= find(any(epo.y==1,1));
end

subidx= cat(2, repmat({':'},[1 ndims(epo.x)-1]), {idx});
epo.x= epo.x(subidx{:});

if isfield(epo, 'y'),
  epo.y= epo.y(:,idx);

  nonvoidClasses= find(any(epo.y==1,2));
  if length(nonvoidClasses)<size(epo.y,1) && opt.removevoidclasses
    msg= sprintf('void classes removed, %d classes remaining', ...
                 length(nonvoidClasses));
    bbci_warning(msg, 'selection', mfilename);
    epo.y= epo.y(nonvoidClasses,:);
    if isfield(epo, 'className'),
      epo.className= {epo.className{nonvoidClasses}};
    end
  end
end

if isfield(epo, 'indexedByEpochs'),
  for Fld= epo.indexedByEpochs,
    fld= Fld{1};
    tmp= getfield(epo, fld);
    subidx= repmat({':'}, 1, ndims(tmp));
    subidx{end}= idx;
    epo= setfield(epo, fld, tmp(subidx{:}));
  end
end

