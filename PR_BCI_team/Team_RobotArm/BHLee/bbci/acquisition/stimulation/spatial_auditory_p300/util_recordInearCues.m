function util_recordInearCues(varargin),

    opt= propertylist2struct(varargin{:});

    opt= set_defaults(opt, ...
        'fs', 44100, ...
        'cueStream', [], ...
        'recordDirections', [], ...
        'filePath', '', ...
        'pahandle', 0, ...
        'userName', 'martijn', ...
        'padding', 0.01);
    
    if isempty(opt.recordDirections),
        opt.recordDirections = opt.speakerSelected;
    end
    
    if isempty(opt.pahandle),
        error('No handle to the sound device has been provided');
    end
    
    if isempty(opt.filePath),
        opt.filePath = pwd; 
    end

    padding = 2*(opt.fs*opt.padding);
    
    for i = opt.recordDirections,
        WaitSecs(.5);
        
        cue = zeros([size(opt.cueStream, 1), size(opt.cueStream, 2)+padding]);
        cue(i,padding/2:(padding/2)+size(opt.cueStream,2)-1) = opt.cueStream(i,:);

        PsychPortAudio('GetAudioData', opt.pahandle, 1);
        PsychPortAudio('FillBuffer', opt.pahandle, cue);

        PsychPortAudio('Start', opt.pahandle, 1, 0, 1);
        PsychPortAudio('Stop', opt.pahandle, 1);
        audiodata = PsychPortAudio('GetAudioData', opt.pahandle);
        wavwrite(audiodata', opt.fs, 16, [opt.filePath '\' opt.userName int2str(i) '.wav']);
        WaitSecs(1);
    end
end