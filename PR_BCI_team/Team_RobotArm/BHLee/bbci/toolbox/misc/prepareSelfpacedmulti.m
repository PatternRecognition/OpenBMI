function prepareSelfpacedmulti(file, filterList, varargin)
%prepareSelfpacedmulti(file, filterList, ...
%                      <classDef>, <displayMontage>, <blockingTime>)

param= varargin;
if length(param)>0 & iscell(param{1}),
  classDef= param{1};
  param= {param{2:end}};
else
  classDef= {70, 74, 66; 'left index', 'right index', 'right foot'};
end
if length(param)>0 & ischar(param{1}),
  displayMontage= param{1};
  param= {param{2:end}};
else
  displayMontage= 'small';
end
if ~iscell(filterList), filterList= {filterList}; end

for il= 1:length(filterList),
  filt= filterList{il};
  if strcmp(filt, 'raw'),
    appendix= '';
  else
    appendix= filt;
  end
  cnt= readGenericEEG(file);
  cnt= calcBipolarChannels(cnt);
  cnt= proc_filtBackForth(cnt, filt);
  mrk= readMarkerTable(file);
%  mrk= fixMultiProblem(mrk);
  mrk= makeClassMarkersMulti(mrk, classDef, param{:});
  mnt= setElectrodeMontage(cnt.clab, displayMontage);
  saveProcessedEEG(file, cnt, mrk, mnt, appendix);
end
