function mrk= mrk_addRunNumbers(mrk, dat)

%MRK_ADDRUNNUMBERS - Adds 'number of run' field to marker structure,
%provided a continuous data struct with a .T field which specifies the
%segment lengths. Alternatively, provide a path to the eeg files.
%
%Usage:
% mrk = mrk_addRunNumbers(mrk,cnt) - extract runs from cnt.T field
% mrk = mrk_addRunNumbers(mrk,file) - extract runs from EEG files
%
% IN:
% file    - path and basename of data files (eg 'VPbla_09_09_09/VPbla_blub')
% cnt     - continuous data containing a .T field 
%
% OUT: Updated marker struct with an additional runNumber field. 

% Benjamin Blankertz, Matthias Treder 2008/2010

if ~isstruct(dat)    % dat refers to eeg file
  % Add wildcard if its not there already
  if isempty(strfind(dat,'*')), dat = [dat '*']; end
  dat=eegfile_readBV(dat,'vars','cnt','clab',1,'fs',mrk.fs);
end

if ~isfield(dat,'T')
  error('Field "T" does not exist.')
end

if isfield(dat,'runNumber')
  warning('Overwriting existing field runNumber')
end

mrk.runNumber = zeros(1,length(mrk.pos));
mrk= mrk_addIndexedField(mrk, 'runNumber');

% Add run numbers
segments = [1 dat.T];
for ii=1:numel(dat.T)
  idx = mrk.pos > sum(segments(1:ii)) & mrk.pos < sum(segments(1:ii+1));
  mrk.runNumber(idx) = ii;
end

