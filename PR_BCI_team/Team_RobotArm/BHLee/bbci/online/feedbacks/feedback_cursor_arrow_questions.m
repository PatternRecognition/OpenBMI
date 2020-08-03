function opt = feedback_cursor_arrow_questions(fig, opt, ctrl);
%FEEDBACK_CURSOR_ARROW_QUESTIONS - BBCI Feedback: 1D Cursor Movement in
%response to questions
%Synopsis:
% OPT= feedback_cursor_arrow(FIG, OPT, CTRL)
%
%Arguments:
% FIG  - handle of figure
% OPT  - struct of optional properties, see below
% CTRL - control signal to be received from the BBCI classifier
%
%Output:
% OPT - updated structure of properties
%
%Optional Properties:
%  duration_show_selected: time to show selection
%  duration_before_free: time after target presentation, before cursor
%    starts moving
%  time_per_trial: maximum duration of a trial
%  timeout_policy: 'miss', 'reject', 'lastposition', 'hitiflateral'
%  rate_control: switch for rate control (opposed to position control)
%  speed: speed (in rate control) 1 means CTRL=1 moves in 1s from
%     center to target
%  cursor_on: switch to show (or hide) cursor
%  response_at: switch to show response (hit vs miss) (1) at 'center' area,
%     or (2) at 'target' position, or (3) in the 'cursor' (cross), or
%     (4) 'none' not at all.
%  trials_per_run: number of trials in one run
%  free_trials: number of free trials (without counting hit or miss)
%  break_every: number of trial after which a break is inserted, 0 means no
%     breaks. Default: 0.
%  break_show_score: 1 means during breaks the score is shown, 0 means not.
%     Default: 1.
%  marker_active_spec: marker specification (cell array) of active cursor
%  marker_inactive_spec: marker specification (cell array) of inactive cursor
%  fixation_spec: marker specification (cell array) of fixation cross
%  background: color of figure background
%  color_hit: color for hit
%  color_miss: color for miss
%  color_reject: color for reject
%  color_target: color to indicate targets
%  color_nontarget: color of non-target
%  center_size: size of center area (for response in center)
%  target_width: width of target areas (normalized)
%  target_dist: distance of targets vertically from top and bottom
%  next_target: switch to show next target
%  next_target_width: width of next target indicator (normalized
%     within target width)
%  msg_spec: text specification (cell array) of messages
%  points_spec: text specification (cell array) of points
%  countdown: length of countdown before application starts [ms]
%  position: position of the figure (pixel)
%  questions: cell string array containing the questions and answers
%  each row must contain a question followed by two answers { q, a1, a2}
%
%Markers written to parallel port
%    1: target on left side
%    2: target on right side
%   11: trial ended with answer 1 on the left side/bottom
%   12: trial ended with answer 1 on the right side/bottom
%   21: trial ended with answer 2 on the left side/bottom
%   22: trial ended with answer 2 on the right side/bottom
%   23: trial ended by time-out and opt.timeout_policy='miss'
%   24: trial ended by time-out and opt.timeout_policy='reject'
%   25: trial ended by time-out and opt.timeout_policy='hitiflateral'
%          and cursor was not lateral enough: reject
%   30: countdown starts
%   31: select goal side and indicate it
%   32: wait before cursor movement starts
%   33: in position control, cursor becomes active in center area
%   34: move cursor until target was hit or time-out
%   35: wait before next trial starts (or game is over)
%   41: first touch with cursor correctly on the left side
%   42: first touch with cursor correctly on the right side
%   51: first touch with cursor erroneously on the left side
%   52: first touch with cursor erroneously on the right side
%   70: start of adaptation phase 1 (rotating cursor)
%   71: end of adaptation phase 1
%  200: init of the feedback
%  210: game status changed to 'play'
%  211: game status changed to 'pause'
%  254: game ends
%
%See:
%  tester_cursor_1d_pro, feedback_cursor_1d_pro_init

% Author(s): Benjamin Blankertz, Nov-2006

persistent H HH state memo cfd

if ~isstruct(opt) | ~isfield(opt,'reset')
    opt.reset = 1;
end

if opt.reset,
    global VP_SCREEN
    opt.reset= 0;
    [opt, isdefault]= ...
        set_defaults(opt, ...
        'countdown', 7000, ...
        'classes', {'left','right'}, ...
        'trigger_classes_list', {}, ...
        'duration', 4000, ...
        'duration_jitter', 0, ...
        'duration_before_free', 1000, ...
        'duration_show_selected', 2000, ...
        'duration_blank', 5000, ...
        'duration_until_hit', 2000, ...
        'duration_break', 15000, ...
        'duration_break_fadeout', 1000, ...
        'duration_break_post_fadeout', 1000, ...
        'touch_terminates_trial', 0, ...
        'break_every', 20, ...
        'break_show_score', 1, ...
        'break_endswith_countdown', 1, ...
        'timeout_policy', 'hitiflateral', ...
        'rate_control', 1, ...
        'cursor_on', 1, ...
        'response_at', 'none', ...
        'trials_per_run',100,...
        'cursor_active_spec', ...
        {'FaceColor',[0.32 0 0.4]}, ...
        'cursor_inactive_spec', ...
        {'FaceColor',[0 0 0]}, ...
        'free_trials', 0, ...
        'background', 0.5*[1 1 1], ...
        'color_hit', [0 0.8 0], ...
        'color_miss', [1 0 0], ...
        'color_reject', [0.8 0 0.8], ...
        'color_center', 0.5*[1 1 1], ...
        'center_size', 0.3, ...
        'damping_in_target', 'quadratic', ...
        'target_width', 0.075, ...
        'frame_color', 0.8*[1 1 1], ...
        'punchline', 0, ...
        'punchline_spec', {'Color',[0 0 0], 'LineWidth',3}, ...
        'punchline_beaten_spec', {'Color',[1 1 0], 'LineWidth',5}, ...
        'gap_to_border', 0.02, ...
        'msg_spec', {'FontSize',0.15}, ...
        'points_spec', {'FontSize',0.065}, ...
        'adapt_trials', 0, ...
        'adapt_duration_rotation', 40*1000, ...
        'adapt_duration_msg', [2 15]*1000, ...
        'adapt_radius', 0.5, ...
        'adapt_rotation_speed', 0.4, ...
        'parPort', 1,...
        'changed', 0,...
        'show_score', 0, ...
        'show_rejected', 0, ...
        'show_bit', 0,...
        'log_state_changes',0, ...
        'verbose', 1, ...
        'log',1,...
        'fs', 25, ...
        'status', 'pause', ...
        'pause_msg', 'pause', ...
        'position', VP_SCREEN,...
        'answer_spec', {'FontSize',0.1},...
        'question_spec', {'FontSize',0.08},...
        'max_characters',22,...
        'cross_offset',-0.2,...
        'cross_visible',1,...
        'question_order', 'ordered',...
        'presentations',5);

    opt.break_every=opt.presentations
    if ~opt.touch_terminates_trial & isdefault.punchline,
        [opt, isdefault]= opt_overrideIfDefault(opt, isdefault, ...
            'punchline', 1);
    end
    if ~ismember(opt.timeout_policy,{'hitiflateral','reject'}) ...
            & isdefault.show_rejected,
        [opt, isdefault]= opt_overrideIfDefault(opt, isdefault, ...
            'show_rejected', 0);
    end
    if opt.adapt_trials>0 & length(opt.free_trials)==1,
        opt.free_trials= [0 opt.free_trials];
    end
    if length(opt.adapt_duration_msg)==1,
        opt.adapt_duration_msg= opt.adapt_duration_msg([1 1]);
    end

    [HH, cfd]= feedback_cursor_arrow_questions_init(fig, opt);
    [handles, H]= fb_handleStruct2Vector(HH);

    do_set('init', handles, 'cursor_arrow', opt);
    do_set(200);

    memo.stopwatch= 0;
    memo.x= 0;
    memo.degree= 0;
    memo.lastdigit= NaN;
    memo.laststate= NaN;
    memo.laststatus= 'urknall';
    memo.timer= 0;
    memo.presentations=0;
    if opt.adapt_trials>0,
        memo.modus= 3;
    else
        memo.modus= 1;
    end
    state= -1;
end

if opt.changed,
    if ~strcmp(opt.status, memo.laststatus),
        memo.laststatus= opt.status;
        switch(opt.status),
            case 'play',
                do_set(210);
                %      if opt.rate_control & opt.cursor_on,
                %        do_set(H.cursor, 'Visible','off');
                %      end
                %      do_set(H.fixation, 'Visible','off');
                do_set(H.msg_punch, 'Visible','off');
                state= 0;
            case 'pause',
                do_set(211);
                do_set(H.msg, 'String',opt.pause_msg, 'Visible','on');
                state= -1;
            case 'stop';
                do_set(254);
                do_set(H.msg, 'String','stopped', 'Visible','on');
                state= -2;
        end
    end
end

if opt.changed==1,
    opt.changed= 0;
    do_set(H.msg, opt.msg_spec{:});
    do_set([H.points H.msg_punch], opt.points_spec{:});
    if opt.rate_control & ~strcmpi(opt.timeout_policy,'hitiflateral'),
        memo.center_visible= 0;
        do_set(H.center, 'Visible','off');
    else
        memo.center_visible= 1;
        do_set(H.center, 'FaceColor',opt.color_center, 'Visible','on');
    end
    if opt.show_score,
        do_set(H.points, 'Visible','on');
    else
        do_set(H.points, 'Visible','off');
    end
    if opt.show_rejected,
        do_set(H.rejected_counter, 'Visible','on');
    else
        do_set(H.rejected_counter, 'Visible','off');
    end
    if opt.punchline,
        do_set(H.punchline, 'Visible','on'),
    else
        do_set(H.punchline, 'Visible','off'),
    end
    if ~opt.cursor_on & strcmp(opt.response_at, 'cursor'),
        opt.response_at= 'center';
    end
end

if state~=memo.laststate,
    memo.laststate= state;
    if state>=0 & opt.log_state_changes,
        do_set(30+state);
    end
    if opt.verbose>1,
        fprintf('state: %d\n', state);
    end
end

if ~opt.rate_control & state>-2,
    memo.x= ctrl;
    memo.x= max(-1, min(1, memo.x));
end

if state==0,
    if memo.timer == 0,
        do_set(H.fixation, 'Visible','off');
    end
    digit= ceil((opt.countdown-memo.timer)/1000);
    if digit~=memo.lastdigit,
        do_set(H.msg, 'String',int2str(digit), 'Visible','on');
        memo.lastdigit= digit;
    end
    if memo.timer>=opt.countdown,
        %% countdown terminates: prepare run
        do_set(H.msg, 'Visible','off');
        memo.timer= 0;
        if memo.modus==0,      %% during break inbetween run
            state= 1;
        else
            memo.punch= [-1 1];
            memo.isyes= [];
            memo.rejected= 0;
            if memo.modus==3,  %% rotation adaptation coming up
                do_set(70);
                state= 20;
            else
                if memo.modus==2,    %% bias adaptation coming up
                    nTrials= 2*opt.adapt_trials;
                else                 %% start of regular feedback part
                    nTrials= size(opt.questions,1)*opt.presentations;
                end
                nBlocks= size(opt.questions,1);
                seq= [];
                for bb= 1:nBlocks,
                    bseq= linspace(1, 2, opt.presentations);
                    middle=find(bseq==0.5);
                    bseq(middle)=bseq(middle)-0.5+rand();
                    bseq=round(bseq);
                    if mod(opt.break_every,2) && mod(bb,2),
                        bseq(end)= 1;
                    end
                    pp= randperm(opt.presentations);
                    seq= cat(2, seq, bseq(pp));
                end
                nLeftOvers= nTrials - length(seq);
                bseq= round(linspace(1, 2, nLeftOvers));
                pp= randperm(nLeftOvers);
                seq= cat(2, seq, bseq(pp));

                if strcmp(opt.question_order,'random'),
                    questions=randperm(size(opt.questions,1));
                elseif strcmp(opt.question_order,'ordered'),
                    questions=[1:size(opt.questions,1)];
                end


                memo.question_sequence=[];
                for i=1:length(questions),
                    memo.question_sequence= [memo.question_sequence ,ones(1,opt.presentations)*questions(i)];
                end

                memo.sequence= [seq, 0];
                memo.trial= -opt.free_trials(memo.modus);
                memo.touched_once= 0;
                state= 1;
            end
        end
    else
        memo.timer= memo.timer + 1000/opt.fs;
    end
end

if state==1,   %% show a blank screen with fixation cross
    if memo.timer == 0,
        do_set(H.cue, 'Visible','off');
        do_set(H.answers, 'Visible','off');
        do_set(H.question, 'Visible','off');
        do_set(H.cursor, 'Visible','off');
        if opt.cross_visible,
            do_set(H.fixation, 'Visible','on');
        end
        fontsize=opt.question_spec{2};
        question=opt.questions{memo.question_sequence(memo.trial+1),1};
        if length(question) > opt.max_characters,
            question={question};
            toolong=1;
            while toolong,
                toolong=0;
                if length(question{end})>opt.max_characters,
                    toolong=1;
                    lastrow=question{end};
                    spaces=strfind(lastrow,' ');
                    space=spaces(find(spaces<opt.max_characters,1,'last'));
                    question{end}=lastrow(1:space);
                    question=[question,lastrow(space+1:end)];
                    fontsize=fontsize*0.9;
                end
            end
        end


        do_set(H.question,'String',question)
        do_set(H.question,'FontSize',fontsize)
        do_set(H.question,'Visible','on');
    end
    if memo.timer >= opt.duration_blank,
        memo.timer= 0;
        state= 2;
    end
    memo.timer= memo.timer + 1000/opt.fs;
end

if state==2,  %% select sides and display answers and question
    memo.trial_duration= opt.duration + opt.duration_jitter*rand;
    memo.trial= memo.trial + 1;
    memo.presentations=memo.presentations+1;
    if memo.trial==1,
        memo.stopwatch= 0;
    end
    if memo.trial<1,
        memo.goal= ceil(2*rand);
    else
        memo.goal= memo.sequence(memo.trial);
    end

    if memo.goal==1,
        answers=[1,2];
    else
        answers=[2,1];
    end
    for i=1:2,
        do_set(H.answers(answers(i)),'Visible','on')
        do_set(H.answers(answers(i)),'String',opt.questions{memo.question_sequence(memo.trial),i+1})
    end
    % chop the question in to lines if necessary


    memo.x= 0;
    memo.timer= 0;
    state= 3;
    if isempty(opt.trigger_classes_list),
        triggi= memo.goal;
    else
        cued_class= opt.classes{memo.goal};
        triggi= strmatch(cued_class, opt.trigger_classes_list);
        if isempty(triggi),
            warning('cued class not found in trigger_classes_list');
            triggi= 10+memo.goal;
        end
    end
    do_set(triggi);
end

if state==3,  %% wait before cursor movement starts
    if memo.timer>=opt.duration_before_free,
        state= 4;
    else
        memo.timer= memo.timer + 1000/opt.fs;
    end
end

if state==4,  %% in position control, cursor becomes active in center area
    if opt.rate_control | abs(memo.x)<opt.center_size,
        do_set(60);
        if opt.cursor_on,
            do_set(H.fixation, 'Visible','off');
            ud= get(HH.cursor, 'UserData');
            if opt.cross_visible
                do_set(H.cursor, 'Visible','on', ...
                    opt.cursor_active_spec{:}, ...
                    'XData',ud.xData, 'YData',ud.yData);
            end
        end
        memo.trialwatch= 0;
        state= 5;
    end
end

if state==5,  %% move cursor until target was hit or time-out
    if opt.rate_control,
        if abs(memo.x)<1,
            memo.x= memo.x + ctrl/opt.fs/opt.duration_until_hit*1000;
        else
            switch(opt.damping_in_target),
                case 'linear',
                    sec= (memo.trial_duration-opt.duration_until_hit)/1000;
                    frames= 1 + ceil( opt.fs * sec );
                    memo.x= memo.x + ctrl * cfd.target_width / frames;
                case 'quadratic',
                    slope= 1000/opt.duration_until_hit;
                    yy= 1 - (abs(memo.x)-1)/cfd.target_width;
                    fac= yy*slope;
                    memo.x= memo.x + ctrl*fac/opt.fs;
                otherwise
                    error('option for damping_in_target unknown');
            end
        end
        %    memo.x= max(-1-cfd.target_width, min(1+cfd.target_width, memo.x));
    end

    trial_terminates= 0;

    %% cursor touches target
    if abs(memo.x) >= 1,
        if opt.touch_terminates_trial,
            trial_terminates= 1;
            memo.selected= sign(memo.x)/2 + 1.5;
            isyes= (memo.selected==memo.goal);
            do_set(10*(2-iyes)+memo.selected);
        elseif ~memo.touched_once,
            memo.touched_once= 1;
            memo.preselected= sign(memo.x)/2 + 1.5;
            isyes= (memo.preselected==memo.goal);
            do_set(30+10*(2-isyes)+memo.preselected);
        end
    end

    %% timeout
    if memo.trialwatch >= memo.trial_duration,
        trial_terminates= 1;
        switch(lower(opt.timeout_policy)),
            case 'miss',
                memo.selected= [];
                iyes= 0;
                do_set(23);  %% reject by timeout
            case 'reject',
                memo.selected= [];
                isyes= -1;
                do_set(24);  %% reject by timeout
            case 'lastposition',
                memo.selected= sign(memo.x)/2 + 1.5;
                isyes= (memo.selected==memo.goal);
                do_set(10*(2-isyes)+memo.selected);
            case 'hitiflateral',
                if abs(memo.x)>opt.center_size,
                    %% cursor is lateral enough (outside center): count as hit
                    memo.selected= sign(memo.x)/2 + 1.5;
                    isyes= (memo.selected==memo.goal);

                    do_set(10*(2-isyes)+memo.selected);
                else
                    %% cursor is within center area: count as reject
                    memo.selected= [];
                    isyes= -1;
                    do_set(25);  %% reject by timeout and position
                end
            otherwise
                error('unknown value for OPT.timeout_policy');
        end
    end

    %% trial terminates: show response and update score
    if trial_terminates,
        do_set(H.cursor, opt.cursor_inactive_spec{:});
        if ~opt.cross_visible
            do_set(H.cursor,'Visible','on')
        end
        if memo.trial>0,
            if isyes==-1,
                memo.rejected= memo.rejected + 1;
            end
            memo.isyes(memo.trial)= (isyes==1);
            nYes= sum(memo.isyes(memo.trial-memo.presentations+1:memo.trial));
            nNo= memo.presentations-nYes;
            if opt.verbose,
                fprintf('%s: %03d %s: %03d  (%d rejected)\n', ...
                    opt.questions{memo.question_sequence(memo.trial),2}, nYes,...
                    opt.questions{memo.question_sequence(memo.trial),3}, nNo, ...
                    memo.rejected);
            end
            % do_set(H.points(1), 'String',['hit: ' int2str(nHits)]);
            %do_set(H.points(2), 'String',['miss: ' int2str(nMisses)]);
            %do_set(H.rejected_counter, 'String',['rej: ' int2str(memo.rejected)]);
            if isyes==1 & ~opt.touch_terminates_trial,
                ii= memo.selected;
                if abs(memo.x) > abs(memo.punch(ii)),  %% beaten the punchline?
                    memo.punch(ii)= memo.x;
                    do_set(H.punchline(ii), 'XData',[1 1]*memo.x, ...
                        opt.punchline_beaten_spec{:});
                end
            end
        end
        switch(lower(opt.response_at)),
            case 'center',
                memo.H_indicator= H.center;
                ind_prop= 'FaceColor';
                do_set(H.center, 'Visible','on');
            case 'target',
                memo.H_indicator= H.cue(memo.goal);
                ind_prop= 'FaceColor';
            case 'cursor',
                memo.H_indicator= H.cursor;
                ind_prop= 'Color';
            case 'none',
                memo.H_indicator= [];
            otherwise,
                warning('value for property ''response at'' unrecognized.');
                memo.H_indicator= H.target(memo.selected);
        end
        if ~isempty(memo.H_indicator),
            switch(isyes),
                case 1,
                    do_set(memo.H_indicator, ind_prop,opt.color_hit);
                case 0,
                    do_set(memo.H_indicator, ind_prop,opt.color_miss);
                case -1,
                    do_set(memo.H_indicator, ind_prop,opt.color_reject);
            end
        end
        memo.timer= 0;
        state= 6;
    end
    memo.trialwatch= memo.trialwatch + 1000/opt.fs;
end

if state==6,  %% wait before next trial starts (or game is over)
    if memo.timer >= opt.duration_show_selected,
        switch(lower(opt.response_at)),
            case 'center',
                if memo.center_visible,
                    do_set(H.center, 'FaceColor',opt.color_center);
                else
                    do_set(H.center, 'Visible','off');
                end
            case 'target',
            case 'cursor',
            case 'none',
        end
        do_set(H.punchline, opt.punchline_spec{:});
        if memo.trial==length(memo.sequence)-1,
            if memo.modus==2,
                %% end of bias adaptation period
                memo.timer= 0;
                state= 21;
            else
                %% game over
                
                if opt.show_bit,
                    minutes= memo.stopwatch/1000/60;
                    acc= sum(memo.isyes)/(memo.trial-memo.rejected);
                    bpm= bitrate(acc) * opt.trials_per_run / minutes;
                    msg= sprintf('%.1f bits/min', bpm);
                    do_set(H.msg, 'String',msg, 'Visible','on');
                else
                  nYes= sum(memo.isyes(memo.trial-memo.presentations+1:memo.trial));
                  nNo=memo.presentations-nYes;
                  %msg= sprintf('%s: %d %s: %d ',opt.questions{memo.question_sequence(memo.trial),2}, nYes,...
                  % opt.questions{memo.question_sequence(memo.trial),3}, nNo);
                  if nYes>nNo,
                    msg= sprintf('%s (%d/%d)',opt.questions{memo.question_sequence(memo.trial),2}, nYes, nNo);
                  else
                    msg= sprintf('%s (%d/%d)',opt.questions{memo.question_sequence(memo.trial),3}, nNo, nYes);
                  end

                  
                  msg= [msg;{' '};{sprintf('Vielen Dank')}];
                  do_set(H.msg, 'String',msg, 'Visible','on');
                end
                if 0, %opt.punchline,
                    msg= sprintf('punch at  [%d %d]', ...
                        round(100*(memo.punch-sign(memo.punch))/cfd.target_width));
                    do_set(H.msg_punch, 'String',msg, 'Visible','on');
                end
                if opt.rate_control,
                    do_set(H.cursor, 'Visible','off');
                else
                    do_set(H.cursor, opt.cursor_inactive_spec{:});
                end
                do_set(H.fixation, 'Visible','off');
                do_set(H.cue, 'Visible','off');
                do_set(H.answers, 'Visible','off');
                do_set(H.question, 'Visible','off');
                do_set(254);
                %       memo.timer= 0;
                %       state= 10;
                state= -1;
            end
        elseif opt.break_every>0 & memo.trial>0 & mod(memo.trial,opt.break_every)==0,
            memo.timer= 0;
            state= 7;
        else
            memo.timer= 0;
            state= 1;
            opt = feedback_cursor_arrow_questions(fig, opt, ctrl);
            return;
        end
    else
        memo.timer= memo.timer + 1000/opt.fs;
    end
end

if state==7,   %% give a break where the score is (optionally) shown
    if memo.timer == 0,
        if opt.break_show_score,
            nYes= sum(memo.isyes(memo.trial-memo.presentations+1:memo.trial));
            nNo=memo.presentations-nYes
            %msg= sprintf('%s: %d %s: %d ',opt.questions{memo.question_sequence(memo.trial),2}, nYes,...
               % opt.questions{memo.question_sequence(memo.trial),3}, nNo);
            if nYes>nNo,
              msg= sprintf('%s (%d/%d)',opt.questions{memo.question_sequence(memo.trial),2}, nYes, nNo);
            else
              msg= sprintf('%s (%d/%d)',opt.questions{memo.question_sequence(memo.trial),3}, nNo, nYes);
            end

            do_set(H.msg, 'String',msg, 'Visible','on');
        end
        memo.presentations=0;
        do_set(H.center, 'Visible','off');
        do_set(H.cursor, 'Visible','off');
        do_set(H.fixation, 'Visible','off');
        do_set(H.cue, 'Visible','off');
        do_set(H.answers, 'Visible','off');
        do_set(H.question, 'Visible','off');
    end
    if memo.timer >= opt.duration_break,
        memo.timer= 0;
        state= 8;
    end
    memo.timer= memo.timer + 1000/opt.fs;
end

if state==8,   %% score fade-out at the end of the break
    if memo.timer>opt.duration_break_fadeout+opt.duration_break_post_fadeout,
        memo.timer= 0;
        if opt.break_endswith_countdown,
            memo.modus= 0;
            state= 0;
        else
            state= 1;
        end
    elseif memo.timer<=opt.duration_break_fadeout,
        fade= (opt.duration_break_fadeout-memo.timer)/opt.duration_break_fadeout;
        do_set(H.msg, 'Color',[0 0 0]*fade + opt.background*(1-fade));
        memo.fadeout_finished= 0;
    else
        if ~memo.fadeout_finished,
            memo.fadeout_finished= 1;
            if memo.center_visible,
                do_set(H.center, 'Visible','on');
            end
            do_set(H.msg, 'Visible','off', 'Color',[0 0 0]);
        end
    end
    memo.timer= memo.timer + 1000/opt.fs;
end

%if state==10,   %% wait before stop recording
%  if memo.timer>=500,
%    bvr_sendcommand('stoprecording');
%    fprintf('EEG recording stopped.\n');
%    state= -1;
%  end
%  memo.timer= memo.timer + 1000/opt.fs;
%end

if state==20,  %% circular moving cross for adaption
    memo.degree= memo.degree + opt.adapt_rotation_speed*2*pi/opt.fs;
    memo.x = opt.adapt_radius*cos(memo.degree);
    memo.y = opt.adapt_radius*sin(memo.degree);
    ud= get(HH.cursor, 'UserData');
    do_set(H.cursor, ...
        'XData',memo.x+ud.xData, ...
        'YData',memo.y+ud.yData, ...
        'Visible','on');
    if memo.timer >= opt.adapt_duration_rotation,
        do_set(71);
        memo.timer= 0;
        state= 21;
    else
        memo.timer = memo.timer + 1000/opt.fs;
    end
end

if state==21,
    if memo.timer==0,
        do_set([H.cue, H.cursor], 'Visible','off');
        do_set(H.msg, 'String','Adapting ...', 'Visible','on');
        do_set(H.points(1), 'String','hit: 0');
        do_set(H.points(2), 'String','miss: 0');
        do_set(H.rejected_counter, 'String','rej: 0');
    end
    if memo.timer >= opt.adapt_duration_msg(4-memo.modus),
        memo.modus= memo.modus - 1;
        memo.timer= 0;
        memo.rejected= 0;
        state= 0;
    else
        memo.timer = memo.timer + 1000/opt.fs;
    end
end

memo.stopwatch= memo.stopwatch + 1000/opt.fs;

if state~=20,
    ud= get(HH.cursor, 'UserData');
    if memo.x<0,
        iDir= 1;
    else
        iDir= 2;
    end
    switch(opt.classes{iDir}),
        case 'left',
            do_set(H.cursor, 'XData',ud.xData - abs(memo.x), 'YData',ud.yData);
        case 'right',
            do_set(H.cursor, 'XData',ud.xData + abs(memo.x), 'YData',ud.yData);
        case {'down','foot'},
            do_set(H.cursor, 'YData',ud.yData - abs(memo.x), 'XData',ud.xData);
        case {'up','tongue'},
            do_set(H.cursor, 'YData',ud.yData + abs(memo.x), 'XData',ud.xData);
        otherwise,
            error(sprintf('unknown value for opt.classes: <%s>', ...
                opt.classes{iDir}));
    end
end

do_set('+');
