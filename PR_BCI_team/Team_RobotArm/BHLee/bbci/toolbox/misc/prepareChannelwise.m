function prepareChannelwise(file, proc, varargin)
%prepareSelfpaced(file, proc, ...
%                 <classDef>, <displayMontage>, <blockingTime>)
%
% note:
%  if the file name contains 'multi' as substring 'makeClassMarkersMulti'
%  is called instead of 'makeClassMarkers'.

param= varargin;
if length(param)>0 & iscell(param{1}),
  classDef= param{1};
  param= {param{2:end}};
else
  classDef= {[65 70],[74 192]; 'left', 'right'};
end
if length(param)>0 & ischar(param{1}),
  displayMontage= param{1};
  param= {param{2:end}};
else
  displayMontage= 'small';
end

cnt= readChannelwiseProcessed(file, proc);
mrk= readMarkerTable(file);
if isempty(strfind(file, 'multi')),
  mrk= makeClassMarkers(mrk, classDef, param{:});
else
  mrk= makeClassMarkersMulti(mrk, classDef, param{:});
end
mnt= setElectrodeMontage(cnt.clab, displayMontage);

saveProcessedEEG(file, cnt, mrk, mnt, proc.appendix);
