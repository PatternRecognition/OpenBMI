function data = bbci_load(bbci)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

bbci= bbci_calibrate_setDefaults(bbci);
BC= bbci.calibrate;

% fullfile does not work for CELL, so we use strcat here:
if ~isempty(BC.folder), % if empty, assume absolute paths
    data.filename= strcat(BC.folder, filesep, BC.file);
else
    data.filename = BC.file;
end
filespec= strcat(data.filename,'.*');
if iscell(filespec),
    data.fileinfo= cellfun(@dir, filespec, 'UniformOutput',false);
else
    data.fileinfo= {dir(filespec)};
end
if isempty(BC.montage_fcn),
    [data.cnt, data.mrk, data.mnt]= ...
        BC.read_fcn(data.filename, BC.read_param{:});
else
    [data.cnt, data.mrk]= BC.read_fcn(data.filename, BC.read_param{:});
    data.mnt= BC.montage_fcn(data.cnt.clab, BC.montage_param{:});
end
if ~isempty(BC.marker_fcn),
    data.mrk= BC.marker_fcn(data.mrk, BC.marker_param{:});
end
data.isnew= 1;

end

