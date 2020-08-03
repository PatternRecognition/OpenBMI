function prepareArtifacts(file, filterList, displayMontage)
%prepareArte(file, filterList, displayMontage)
%
% file= 'Gabriel_01_07_24/selfpaced1sGabriel';
% prepareArte(file, {'raw','cut50'});

if exist('displayMontage') & ~isempty(displayMontage)
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
  mrk= readMarkerTableArtifacts(file);
  mrk= makeClassMarkersArtifacts(mrk);
  mnt= setElectrodeMontage(cnt.clab, displayMontage);
  saveProcessedEEG(file, cnt, mrk, mnt, appendix);
end
