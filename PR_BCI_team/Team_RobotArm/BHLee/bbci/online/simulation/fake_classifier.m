function fake_classifier(varargin)
% fake_classifier(<opts>)
%
% fake_classifier simulates an online setting in a controlled manner. 
% Particularly useful for p300 settings, where each marker corresponds
% to exactly 1 classification score.
%
% IN opt:
%     target - the marker number(s) that is assigned target (scalar/vector)
%     val_mrk - vector of all valid marker numbers that should be
%               considered
%     end_mrk - marker that will shut down the routine
%     response_delay - Simulated classifier timewindow 
%     weights - vector with the classifier scores to be send [target non-target]
%     noise - noise to be added to the weights (uses as randn*opt.noise)
%     pyff_port - port to send the scores to
%     pyff_machine - machine to send the scores to
%     bv_machine - machine where BrainVision is running
%
%
%     Particular case (photobrowser)
%     subtrial_mrk - In case a single marker defines the start of the
%                       subtrial and is followed by markers that define
%                       the highlights within that subtrial, the start
%                       marker should be given here.
%     mrk_per_subtr - In case subtrial_mrk is set, it expects this amount
%                     of markers after the subtrial_mrk


%
% OUT  (over udp)
%     send_xml_udp('i:controlnumber', block, 'timestamp', timestamp, ...
%                       'cl_output', [classifierscore marker]);     
%
%
% Example
%     fake_classifier('target', [3 5], 'val_mrk', [1:40], 'response_delay',
%                       1000, 'noise', .3);
%     
                
%
% Martijn Schreuder, Aug 2010



global general_port_fields parport_on acquire_func

opt= propertylist2struct(varargin{:});

[opt, isdefault]= ...
    set_defaults(opt, ...
                 'target', [1 3], ...
                 'val_mrk', [1:10], ...
                 'end_mrk', 254, ...                 
                 'response_delay', 800, ...     
                 'weights', [-1 1], ... %[target nontarget]
                 'noise', 0, ...                 
                 'pyff_port', 12345, ...
                 'pyff_machine', '127.0.0.1', ...
                 'bvmachine', '127.0.0.1', ...
                 'cycle_pause', .0, ...
                 'fs', 100, ...
                 'subtrial_mrk', [], ...
                 'mrk_per_subtr', 1, ...
                 'pause_length', 2000, ...
                 'fb_type', 'pyff');

if isempty(parport_on), parport_on = 1;end
if isempty(opt.subtrial_mrk),
    opt.mrk_per_subtr = 1;
    opt.subtrial_mrk = NaN;
end

acquire_state = acquire_func(opt.fs,opt.bvmachine);
acquire_state.reconnect= 1;

send_xml_udp('init', opt.pyff_machine, opt.pyff_port);

mrk_loc_buff = ones(1,opt.mrk_per_subtr)*NaN;
mrk_buff = ones(1,10)*NaN;
ts_buff = ones(1,10)*NaN;
class_buff = ones(1,10)*NaN;

run = 1;
to_stop = 0;
eval_cue = 0;
resp_counter = 1;
count = 1;
tic;

while run,
    [currData, block, markerPos, markerToken, markerDescr] = ...
        acquire_func(acquire_state);

    timestamp = block*1000/opt.fs;
     
    % TRANSFER MARKERPOS CORRECTLY
    markerPos = markerPos-size(currData,1)+1;
    markerPos = markerPos*1000/opt.fs;
    
    if ~isempty(markerToken),
        if toc*1000 > opt.pause_length,
            mrk_loc_buff = ones(1,opt.mrk_per_subtr)*NaN;
            mrk_buff = ones(1,10)*NaN;
            ts_buff = ones(1,10)*NaN;
            class_buff = ones(1,10)*NaN;
            count = 1;
            tic;
        end
        
        for mrkI = 1:length(markerToken),
            curMrk = str2num(markerToken{mrkI}(2:end));
            
            % set flag for graceful closing
            if curMrk == opt.end_mrk,
                to_stop = 1;
                eval_cue = 1;
            end
            
            % add marker value to the cue
            if ismember(curMrk, opt.val_mrk),
%                 fprintf('\nReceived: marker %i', curMrk);
                mrk_loc_buff(find(isnan(mrk_loc_buff), 1)) = curMrk;
                if isnan(opt.subtrial_mrk),
                    mrk_buff(find(isnan(mrk_buff), 1)) = curMrk;
                    ts_buff(find(isnan(ts_buff), 1)) = timestamp;
                    eval_cue = 1;
                elseif ~any(isnan(mrk_loc_buff)),
                    eval_cue = 1;
                end
            end
            
            % start new 'block-subtrial'
            if curMrk == opt.subtrial_mrk,
                ts_buff(find(isnan(ts_buff), 1)) = timestamp;
                mrk_buff(find(isnan(mrk_buff), 1)) = opt.subtrial_mrk;
                eval_cue = 1;
            end
            
            % calculate classifier value
            if eval_cue,
                % calculate old trial if not first trial
                if ~isnan(mrk_buff(1)) && ~sum(isnan(mrk_loc_buff)),
                    disp(sprintf('\nNr %i: [%s] -> %0.2g', count, num2str(mrk_loc_buff), opt.weights(~any(ismember(opt.target, mrk_loc_buff))+1)));
                    class_buff(find(isnan(class_buff), 1)) = opt.weights(~any(ismember(opt.target, mrk_loc_buff))+1)+randn*opt.noise;
                    count = count+1;tic;
                end
            
                % reset buffer
                mrk_loc_buff = ones(1,opt.mrk_per_subtr)*NaN;
                eval_cue = 0;
            end
        end
    end

    % send value if delay reached
    if timestamp > ts_buff(1)+opt.response_delay && ~isnan(class_buff(1)),
        send_xml_udp('i:controlnumber', block, ...
                       'timestamp', timestamp, ...
                       'cl_output', [class_buff(1) mrk_buff(1)]);
        
%         fprintf('\nSend %i: marker %i - class. %0.2g', resp_counter, mrk_buff(1), class_buff(1));
        resp_counter = resp_counter+1;
        class_buff = circshift(class_buff, [1 -1]); class_buff(end) = NaN;
        ts_buff = circshift(ts_buff, [1 -1]); ts_buff(end) = NaN;
        mrk_buff = circshift(mrk_buff, [1 -1]); mrk_buff(end) = NaN;
    end
    
    % stop procedure
    if to_stop && isnan(class_buff(1)),
        run = 0;
    end
    
    pause(opt.cycle_pause);
end

acquire_bv('close');
send_xml_udp('close')

end
