function [A_th_db, A_th, db_values] = determine_hearing_threshold_via_staircase(varargin)


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt ...
                  ,'test_tone', [] ...
                  ,'fs', 44100 ... % sampling rate of test_tone
                  ,'db_start', -30 ... % start value for threshold procedure
                  ,'delta_dB', 2 ...% step width in the stair-case procedure
                  ); 

if isempty(opt.test_tone)
    opt.test_tone = stimutil_generateTone(500, 'harmonics',7, 'duration', 2 * 1000, 'pan', 1, 'fs', opt.fs, 'rampon', 20, 'rampoff', 50)';
end

db_values = opt.db_start;
counter = 1;
stop = false;
delta_dB = 2; % step width in in/de-creasing the volume
direction = -1; % decrease volume in the beginning
while not(stop)
    
    % play the sound at current amplitude
    A_tmp_db = db_values(counter);
    A_tmp = 10^(A_tmp_db/20);
    wavplay(A_tmp*opt.test_tone', opt.fs, 'async')
    
    % collect response
    display('Was the sound perceived?')
    response = [];
    while not( strcmpi(response, 'yes') || strcmpi(response, 'no') || strcmpi(response, 'stop'))
        response = input('Type <yes>/<no> or <stop> to end the procedure: ', 's');
    end
    % determine whether to increase or decrease amplitude
    switch lower(response)
        case 'yes'
            % if we were approaching the threshold from below, we have 
            % surpassed it. thus start decreasing the amplitude
            if direction > 0
                direction = -1;
            end
        case 'no'
            % if we were approaching the threshold from above, we have 
            % surpassed it. thus start increasing the amplitude
            if direction < 0
                direction = 1;
            end
        case 'stop'
            stop = true;
    end
    
    % update sound amplitude
    if not(stop)
        db_values(counter+1) = A_tmp_db + direction*delta_dB;
        counter = counter + 1;
    end
end

% compute auditory threshold as average of reversal points
reversal_indices = find(diff(diff(db_values))) + 1;
A_th_db = mean(db_values(reversal_indices));
A_th = 10^(A_th_db/20);