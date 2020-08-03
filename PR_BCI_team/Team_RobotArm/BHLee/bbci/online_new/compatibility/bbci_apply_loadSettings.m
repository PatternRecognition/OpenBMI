function bbci = bbci_apply_loadSettings(file),
%BBCI_APPLY_LOADSETTINGS - Load the settings for bbci_apply from a file. If
%  the file is in the old format, it will be converted to the new format.
%
%Synopsis:
%  BBCI= bbci_apply_loadSettings(FILE)
%
%Arguments:
%  FILE - String containing the location of the file. If file is not an
%  absolute path, EEG_RAW_DIR will be prepended.
%
%Output:
%  BBCI - New bbci setup struct for bbci_apply.

% 02-2011 Martijn Schreuder

global EEG_RAW_DIR

if ~isabsolutepath(file),
    file = strcat(EEG_RAW_DIR, file);
end

if ~exist(file, 'file') & ~exist([file '.mat'], 'file'),
    error(sprintf('Settings: %s could be found', file));
end

settings = load(file);
if isnewstyle(settings),
    bbci = settings.bbci;
else
    warning('Settings file is in old format. Now trying to convert.');
    bbci = bbci_apply_convertSettings(settings);
end

end


function boolEval = isnewstyle(settings),
	if isfield(settings, 'bbci') & isfield(settings.bbci, {'source', ...
            'marker', 'signal', 'feature', 'classifier', 'control', ...
            'feedback'}),
        boolEval = true;
    else
        boolEval = false;
    end
end
