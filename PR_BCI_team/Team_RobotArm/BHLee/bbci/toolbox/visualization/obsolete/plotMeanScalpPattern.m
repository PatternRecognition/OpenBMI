function [h, H]= plotMeanScalpPattern(erp, mnt, ival, varargin)
%[h, H]= plotMeanScalpPattern(erp, mnt, ival, <opts>)

bbci_obsolete(mfilename, 'scalpPattern');

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'class',[]);

eee= proc_selectIval(erp, ival);
if max(sum(eee.y,2))>1,
  eee= proc_average(eee);
end
eee= proc_meanAcrossTime(eee);
head= mnt_adaptMontage(mnt, eee);
eee= proc_selectChannels(eee, head.clab(find(~isnan(head.x))));
head= mnt_adaptMontage(mnt, eee);
if ~isempty(opt.class),
  eee= proc_selectClasses(eee, opt.class);
end

if size(eee.x,3)==1,
  H= plotScalpPattern(head, squeeze(eee.x), opt);
  h= H.ax;
else
  error('not implemented for multi classes so far');
end
