function saveProcessedEEG(file, cnt, mrk, mnt, appendix)
%saveProcessedEEG(file, cnt, mrk, mnt, <appendix>)
%
% IN   file     - file name (relative to EEG_MAT_DIR unless beginning with '/')
%      cnt      - contiuous EEG signals, see readGenericEEG
%      mrk      - class markers, see makeClassMarker
%      mnt      - electrode montrage, see setElectrodeMontage
%      appendix - appendix to file name specifying preprocessing type
%
% SEE  readGenericEEG, makeClassMarker, setElectrodeMontage
%
% GLOBZ  EEG_MAT_DIR

global EEG_MAT_DIR


if file(1)==filesep,
  fullName= file;
else
  fullName= [EEG_MAT_DIR file];
end
if exist('appendix','var') & ~isempty(appendix),
  fullName= [fullName '_' appendix];
end
  
[filepath, filename]= fileparts(fullName);
if ~exist(filepath, 'dir'),
  [parentdir, newdir]=fileparts(filepath);
  [status,msg]= mkdir(parentdir, newdir);
  if status~=1,
    error(msg);
  end
  if isunix,
    unix(sprintf('chmod a-rwx,ug+rwx %s', filepath));
  end
end

save(fullName, 'cnt','mrk','mnt');
