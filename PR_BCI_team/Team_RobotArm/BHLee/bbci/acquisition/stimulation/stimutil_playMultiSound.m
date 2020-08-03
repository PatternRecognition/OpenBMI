function stimutil_playMultiSound(sounds, varargin),

% Synopsis: uses the PsychToolbox PsychPortAudio driver to play a sound
% (or multiple sounds in a cell array) through the indicated speaker.
% With the M-Audio FireWire 410 up to 8 speakers can be individually
% used, or any combination of those.
%
% use:
%    stimutil_playMultiSound(sounds, opt);
%
% INPUT
%    sounds          Cell array with the different sounds that should
%                    be played
%    OPT
%     .interval      Defines the interval between playing the elements
%                    of the 'sounds' cell array. When it is a vector
%                    with length == length(sounds), the interval
%                    between every element is explicitly defined. When
%                    it is a scalar, equal interval for every sound is
%                    assumed [default = 1].
%     .placement     Defines the speaker that should be used. When it 
%                    is a vector with length == length(sounds), the 
%                    used speaker is explicitly defined for every
%                    element of 'sound'. When it is a scalar, the same
%                    speaker is used for every sound. If empty, all
%                    speakers are used. [default = []];
%     .repeat        Number of repeats over the entire sequenze
%     .order         The order in which to iterate over the 'sounds'
%                    sequence. Can be 'normal' [=default] or
%                    'reverse'. When reverse, interval and placement
%                    will also be interpreted in reverse order.
%     .pahandle      The handle to the soundcard opened with
%                    PsychPortAudio
%     .speakerCount  The number of speakers that have been loaded for
%                    the soundcard
%
% NOTE
% opt.pahandle can be obtained with the following functions (if the
% PsychToolbox is installed):
%
% InitializePsychSound(1);
% pahandle = PsychPortAudio('Open' [, deviceid][, mode][,reqlatencyclass][, freq][, channels][, buffersize][, suggestedLatency][, selectchannels]);
%
%
% Martijn Schreuder, 11/08/2009
    
    opt= propertylist2struct(varargin{:});
    opt= set_defaults(opt, ...
        'interval', 1, ...
        'placement', [], ...
        'repeat', 1, ...
        'order', 'normal', ...
        'speakerCount', 8);
    
    if ~iscell(sounds),
        sounds = {sounds};
    end
    if length(opt.interval) == 1 && length(sounds) > 1,
        opt.interval = opt.interval*ones([1, length(sounds)]);
    end
    if ~isempty(opt.placement) && (length(opt.placement) ~= length(sounds)),
        if length(opt.placement) == 1,
            % apply to all elements
            opt.placement = opt.placement*ones([1, length(sounds)]);
        else
            error('If placement is set, it''s length should be equal to the number of sounds.\n');
        end
    end
    
    switch opt.order,
        case 'reverse'
            idx = length(sounds):-1:1;
        otherwise 
            idx = 1:length(sounds);
    end
    for j = 1:opt.repeat,
        for i=idx,
            clear cont;
            if isempty(opt.placement), %%% FOR TESTING AND DEBUGGING
%                 cont = repmat(sounds{i}', opt.speakerCount, 1);
                    cont = sounds{i};
            else
                cont = zeros([opt.speakerCount, length(sounds{i})]);
                cont(opt.placement(idx(i)), :) = sounds{i}';
            end
            
            if size(cont,1) ~= opt.speakerCount & size(cont,2) == opt.speakerCount
                cont = cont';
%                 warning('cont data was transposed')
            end
            PsychPortAudio('FillBuffer', opt.pahandle, cont);
            PsychPortAudio('Start', opt.pahandle);
            PsychPortAudio('Stop', opt.pahandle, 1);
            if (j == opt.repeat && i ~= idx(end)) || j < opt.repeat,
                pause(opt.interval(idx(i)));
            end
        end
    end
end