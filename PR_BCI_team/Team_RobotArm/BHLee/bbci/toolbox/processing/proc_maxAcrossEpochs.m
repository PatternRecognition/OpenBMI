function epo= proc_maxAcrossEpochs(epo, varargin)
%PROC_MAXACROSSEPOCHS
%
%Example:
%  epo_r= proc_r_square_signed(epo, 'multiclass_policy','all-against-last');
%  max_r= proc_maxArossEpochs(epo_r);

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'sign', 0);

if opt.sign~=0,
  error('not implemented');
end

[Xmax,idx]= max(abs(epo.x), [], 3);
for ii= 1:size(epo.x,1),
  for jj= 1:size(epo.x,2),
    Xmax(ii,jj)= epo.x(ii,jj,idx(ii,jj));
  end
end
epo.x= Xmax;
epo.className= {'max'};
epo.y= 1;
