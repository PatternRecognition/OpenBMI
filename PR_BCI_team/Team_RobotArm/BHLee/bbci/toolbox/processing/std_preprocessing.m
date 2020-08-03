function [cnt opt, mrk] = std_preprocessing(cnt, mrk, varargin)
% Some standard preprocessing steps. High- and low-pass filtering and
% artifact removal.
%
% sven.daehne@tu-berlin.de

%% options
opt= propertylist2struct(varargin{:});

% default artifact reject options
tmp_opt = struct(...
    'do_artifact_rejection', 1, ...
    'artifact_ival', [0,800], ...
    'visualize', 0, ...
    'do_multipass', 1, ...
    'do_bandpass', 0, ...
    'whiskerlength', 3.5,...
    'remove_bad_channels', 1, ...
    'remove_bad_trials', 1 ...
    );
opt = set_defaults(opt, tmp_opt);



%% filtering
cnt = filter_bandpass(cnt, opt);


%% artifact rejection
if opt.do_artifact_rejection
    [foo, opt.rclab, opt.rtrials] = ...
        reject_varEventsAndChannels(cnt, mrk, opt.artifact_ival, opt);
    if opt.remove_bad_channels && not(isempty(opt.rclab))
        cnt = proc_selectChannels(cnt, 'not', opt.rclab);
        fprintf('removing %d channels\n', length(opt.rclab));
    end
    if opt.remove_bad_trials && not(isempty(opt.rtrials))
        mrk = mrk_selectEvents(mrk, opt.rtrials, 'invert', 1);
        fprintf('removing %d trials\n', length(opt.rtrials));
    end
end
