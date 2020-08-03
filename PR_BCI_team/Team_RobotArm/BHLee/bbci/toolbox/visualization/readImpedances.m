function [labels, impedances] = readImpedances(file)
%  [labels, impedances] = readImpedances(file)
% IN:  file        string specifying the path of the .vhdr-file relative to the directory EEG_RAW_DIR; 
%                  leave out the ending _0_10_.vhdr or _0_100.vhdr!
%
% OUT: labels      cell array of strings; the names of the electrodes.
%      impedances  double array of impedance values as given in the .vhdr-file.
% 
% example: file = 'Matthias_03_08_29/impedances_after'

% kraulem 09.09.2003

global PLAYER

if isempty(PLAYER)
  PLAYER = 1;
end

% read in the header file with the specified impedances in the "low impedance"-section
fid = fopen([file, '_0_10.vhdr'],'r');
F = fread(fid);
impedancestring = char(F');

% cut off leading characters, up to the title of the impedances:
impedancestring = impedancestring(strfind(impedancestring, 'Impedance'): end);
impedancestring = impedancestring((1+strfind(impedancestring, sprintf('\n'))): end);

% read this string into a cell-array
lines = strread(impedancestring, '%s', 'delimiter', sprintf('\n'));

% convert numbers to their numerical values. If "Out of Range!": convert to NaN.
impedances = ones(1,length(lines));
for ind = 1:length(lines)
  impedanceconv = str2num(lines{ind}((strfind(lines{ind}, ':')+1):end));
  if isempty(impedanceconv)
    impedances(ind) = NaN;
  else
    impedances(ind) = impedanceconv;
  end
  lines{ind} = lines{ind}(1:(strfind(lines{ind}, ':')-1));
end
labels = lines';


% read in the header file with the specified impedances in the "high impedance"-section
fid = fopen([file, '_0_100.vhdr'],'r');
F = fread(fid);
impedancestring = char(F');

% cut off leading characters, up to the title of the impedances:
impedancestring = impedancestring(strfind(impedancestring, 'Impedance'): end);
impedancestring = impedancestring((1+strfind(impedancestring, sprintf('\n'))): end);

% read this string into a cell-array
lines = strread(impedancestring, '%s', 'delimiter', sprintf('\n'));

% convert numbers to their numerical values. If "Out of Range!": leave at NaN.
for ind = 1:length(lines)
  if isnan(impedances(ind))
    impedanceconv = str2num(lines{ind}((strfind(lines{ind}, ':')+1):end));
    if ~isempty(impedanceconv)
      impedances(ind) = impedanceconv;
    end
  end
end



if PLAYER==1
  indic = setdiff(1:length(labels),strmatch('x',labels));
else
  indic = strmatch('x',labels);
end

labels = labels(indic);
impedances = impedances(indic);

if PLAYER==2
  for i = 1:length(labels)
    labels{i} = labels{i}(2:end);
  end
end



return


