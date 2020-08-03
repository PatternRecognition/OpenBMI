function epo= proc_subtractReferenceClass(epo, epo_ref, varargin)
%epo= proc_subtractReferenceClass(epo, epo_ref)
%
% when the reference class is one class of epo, use
% proc_classDifference

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'subtract_from', '*');

clInd= getClassIndices(epo, opt.subtract_from);
nClasses= length(clInd);
global_ref= 1;
if size(epo_ref.x, 3)>1,
  epo_ref= proc_average(epo_ref);
  if size(epo_ref.x, 3)>1,
    if size(epo_ref.x, 3) ~= nClasses,
      error('#epochs in epo_ref must be 1 or equal to #classes in epo.');
    end
    global_ref= 0;
  end
end

ref_class= 1;
idx= find(any(epo.y(clInd,:),1));
nEpochs= length(idx);
for ee= 1:nEpochs,
  ii= idx(ee);
  if ~global_ref,
    ref_class= clInd*epo.y(clInd,ii);
  end
  if isfield(epo, 'yUnit') & isequal(epo.yUnit, 'dB'),
    epo.x(:,:,ii)= epo.x(:,:,ii) ./ epo_ref.x(:,:,ref_class);
  else
    epo.x(:,:,ii)= epo.x(:,:,ii) - epo_ref.x(:,:,ref_class);
  end
end

if global_ref,
  epo.className= strcat(epo.className, '-', epo_ref.className{1});
else
  epo.calssName= strcat(epo.className, '-', epo_ref.className);
end