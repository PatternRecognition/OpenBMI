function [EEG, command, dat] = Emotiv_Load(filename); 
EEG = [];
command = '';

if nargin < 1
	% ask user
	[filename, filepath] = uigetfile('*.*', 'Choose a data file -- pop_biosig()'); %%% this is incorrect in original version!!!!!!!!!!!!!!
    drawnow;
    
	if filename == 0 return; end;
	filename = [filepath filename];
%     
    if length(result) == 0 return; end;
    
    % decode GUI params
    % -----------------
    options = {};
    if ~isempty(result{1}), options = { options{:} 'channels'   eval( [ '[' result{1} ']' ] ) }; end;
    if ~isempty(result{2}), options = { options{:} 'blockrange' eval( [ '[' result{2} ']' ] ) }; end;
    if length(result) > 2
        if ~isempty(result{5}), options = { options{:} 'ref'        eval( [ '[' result{5} ']' ] ) }; end;
        if ~result{3},          options = { options{:} 'importevent' 'off'  }; end;
        if  result{4},          options = { options{:} 'blockepoch'  'off' }; end;
    end;
else
    options = varargin;
end;

% decode imput parameters
% -----------------------
g = finputcheck( options, { 'blockrange'  'integer' [0 Inf]    [];
                            'channels'    'integer' [0 Inf]    [];
                            'ref'         'integer' [0 Inf]    [];
                            'rmeventchan' 'string'  { 'on';'off' } 'on';
                            'importevent' 'string'  { 'on';'off' } 'on';
                            'blockepoch'  'string'  { 'on';'off' } 'off' }, 'pop_biosig');
if isstr(g), error(g); end;

% import data
% -----------
EEG = eeg_emptyset;
if ~isempty(g.channels)
     dat = sopen(filename, 'r', g.channels,'OVERFLOWDETECTION:OFF');
else dat = sopen(filename, 'r', 0,'OVERFLOWDETECTION:OFF');
end
fprintf('Reading data in %s format...\n', dat.TYPE);

if ~isempty(g.blockrange)
    newblockrange    = g.blockrange;
    newblockrange(2) = min(newblockrange(2), dat.NRec);
    newblockrange    = newblockrange*dat.Dur;    
    DAT=sread(dat, newblockrange(2)-newblockrange(1), newblockrange(1));
else 
    DAT=sread(dat, Inf);% this isn't transposed in original!!!!!!!!
    newblockrange    = [];
end
sclose(dat);

if strcmpi(g.blockepoch, 'off')
    dat.NRec = 1;
end;

if ~isempty(newblockrange)
    interval(1) = newblockrange(1) * dat.SampleRate(1) + 1;
    interval(2) = newblockrange(2) * dat.SampleRate(1);
else interval = [];
end
    
EEG =convert_EEGlab(dat, DAT, interval, g.channels, strcmpi(g.importevent, 'on'));

if strcmpi(g.rmeventchan, 'on') & strcmpi(dat.TYPE, 'BDF') & isfield(dat, 'BDF')
    if size(EEG.data,1) >= dat.BDF.Status.Channel, 
        disp('Removing event channel...');
        EEG.data(dat.BDF.Status.Channel,:) = []; 
        if ~isempty(EEG.chanlocs) && length(EEG.chanlocs) >= dat.BDF.Status.Channel
            EEG.chanlocs(dat.BDF.Status.Channel) = [];
        end;
    end;
    EEG.nbchan = size(EEG.data,1);
end;

% rerefencing
% -----------
if ~isempty(g.ref)
    disp('Re-referencing...');
    EEG.data = EEG.data - repmat(mean(EEG.data(g.ref,:),1), [size(EEG.data,1) 1]);
    if length(g.ref) == size(EEG.data,1)
        EEG.ref  = 'averef';
    end;
    if length(g.ref) == 1
        disp([ 'Warning: channel ' int2str(g.ref) ' is now zeroed (but still present in the data)' ]);
    else
        disp([ 'Warning: data matrix rank has decreased through re-referencing' ]);
    end;
end;

% convert data to single if necessary
% -----------------------------------
EEG = eeg_checkset(EEG,'makeur');   % Make EEG.urevent field
EEG = eeg_checkset(EEG);

% history
% -------
if isempty(options)
    command = sprintf('EEG = pop_biosig(''%s'');', filename); 
else
    command = sprintf('EEG = pop_biosig(''%s'', %s);', filename, vararg2str(options)); 
    
    
    
    
    
end;    
