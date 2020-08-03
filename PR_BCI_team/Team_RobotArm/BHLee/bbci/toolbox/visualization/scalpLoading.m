function H= scalpLoading(erp, mnt, ival, varargin)
%SCALPLOADING - Display average trials as scalp topography
%
%Usage:
% H= scalpLoading(ERP, MNT, IVAL, <OPTS>)
%
%Input:
% ERP  - struct of epoched EEG data. For convenience used classwise
%        averaged data, e.g., the result of proc_average.
% MNT  - struct defining an electrode montage
% IVAL - The time interval for which scalp topography is to be plotted.
% OPTS - struct or property/value list of optional fields/properties:
%  .class - specifies the class (name or index) of which the topogaphy
%        is to be plotted. For displaying topographies of several classes
%        use scalpPatterns.
%  The opts struct is passed to scalpPlot.
%
%Output:
% H:     Handle to several graphical objects.
%
%See also scalpPatterns, scalpEvolution, scalpPlot.

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'class',[]);

eee= erp;
if nargin>=3 & ~isempty(ival),
  eee= proc_selectIval(eee, ival);
end
if ~isempty(opt.class),
  eee= proc_selectClasses(eee, opt.class);
end
if max(sum(eee.y,2))>1,
  eee= proc_average(eee);
end
if size(eee.x,3)>1,
  error('For plotting topographies of multiple classes use ''scalpPatterns''');
end
eee= proc_meanAcrossTime(eee);
head= mnt_adaptMontage(mnt, eee);
eee= proc_selectChannels(eee, head.clab(find(~isnan(head.x))));
head= mnt_adaptMontage(mnt, eee);

H= plot_scalp_loading(head, squeeze(eee.x), opt);
if isfield(opt, 'sublabel'),
  yLim= get(gca, 'yLim');
  H.sublabel= text(mean(xlim), yLim(1)-0.04*diff(yLim), opt.sublabel);
  set(H.sublabel, 'verticalAli','top', 'horizontalAli','center', ...
                  'visible','on');
end

if nargout<1,
  clear H;
end
