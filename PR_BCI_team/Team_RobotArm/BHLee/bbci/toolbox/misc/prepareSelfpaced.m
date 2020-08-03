function prepareSelfpaced(file, filterList, varargin)
%prepareSelfpaced(file, filterList, ...
%                 <classDef>, <displayMontage>, <blockingTime>)
%
% file= 'Gabriel_01_07_24/selfpaced1sGabriel';
% prepareSelfpaced(file, {'raw','display'});

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
if ~iscell(filterList), filterList= {filterList}; end

for il= 1:length(filterList),
  filt= filterList{il};
  if isequal(filt, 'raw'),
    appendix= '';
  elseif isequal(filt, 'display'),
    clab= readGenericHeader(file);
    emgChans= chanind(clab, 'EMG*');
    nonEMG= chanind(clab, 'not', 'EMG*');
    filt.appendix= 'display';
    filt.chans= {nonEMG, emgChans};
    filt.eval= {['cnt= proc_filtBackForth(cnt, ''cut50''); ' ...
                 'cnt= proc_jumpingMeans(cnt, 10); '],
                ['cnt= proc_filtBackForth(cnt, ''emg''); ' ...
                 'cnt= proc_filtNotchbyFFT(cnt); ' ...
                 'cnt= proc_rectifyChannels(cnt); ' ...
                 'cnt= proc_jumpingMeans(cnt, 10); ']};
  end
  if isstruct(filt),
    appendix= filt.appendix;
    cnt= readChannelwiseProcessed(file, filt);
  else
    cnt= readGenericEEG(file);
    cnt= calcBipolarChannels(cnt);
    cnt= proc_filtBackForth(cnt, filt);
  end
  mrk= readMarkerTable(file);
  mrk= makeClassMarkers(mrk, classDef, param{:});
  mnt= setElectrodeMontage(cnt.clab, displayMontage);
  mnt= excenterNonEEGchans(mnt);  
  saveProcessedEEG(file, cnt, mrk, mnt, appendix);
end
