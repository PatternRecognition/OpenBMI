function [H, Ctour]= scalpPattern(erp, mnt, ival, varargin)
%SCALPPATTERN - Display average trials as scalp topography
%
%Usage:
% H= scalpPattern(ERP, MNT, IVAL, <OPTS>)
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
% Ctour: Struct of contour information
%
%See also scalpPatterns, scalpEvolution, scalpPlot.

opt= propertylist2struct(varargin{:});
[opt, isdefault]= set_defaults(opt, ...
                               'class', [], ...
                               'yUnit', '', ...
                               'contour', 0);
if isdefault.yUnit && isfield(erp, 'yUnit'),
  opt.yUnit= ['[' erp.yUnit ']'];
end

eee= erp;
if nargin>=3 & ~isempty(ival) & ~any(isnan(ival)),
  eee= proc_selectIval(eee, ival, 'ival_policy','minimal');
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

if opt.contour,
    [H, Ctour]= scalpPlot(head, squeeze(eee.x), opt);
else
    H = scalpPlot(head, squeeze(eee.x), opt);
end

if isfield(opt, 'sublabel'),
  yLim= get(gca, 'yLim');
  H.sublabel= text(mean(xlim), yLim(1)-0.04*diff(yLim), opt.sublabel);
  set(H.sublabel, 'verticalAli','top', 'horizontalAli','center', ...
                  'visible','on');
end

if nargout<2,
  clear Ctour;
end
if nargout<1,
  clear H;
end
