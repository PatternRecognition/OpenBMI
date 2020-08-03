function mrk = mrk_addnokeypress(mrk_init, opt)
    % create a nokeypress-marker for each target-marker not associated with a keypress
    % constraint: nokeypress-marker is before the keypress-marker during the trial

    if ~exist('opt', 'var') opt = []; end
    
    opt = set_defaults(opt, 'precision', [-100 100], ... % keypress precision tolerance (in ms)
                            'cls_ival', [-300 -200], ... % classification interval
                            ...                          % of the online feedback (in ms)
                            'response_marker', 'R 16', ...
                            'cls_kp_marker', 'KP', ...
                            'cls_nokp_marker', 'noKP', ...
                            'trialstart_marker', 'S 40', ...
                            'trialend_marker', {'S 41', 'S 42'}, ...
                            'target_markers', {'S  1', 'S  9', 'S 18', 'S 27'});

    precision = opt.precision * (mrk_init.fs/1000);
    cls_ival = opt.cls_ival * (mrk_init.fs/1000);
    start =  min(find(strcmpi(mrk_init.desc, opt.trialstart_marker)));
    for m = 1:length(opt.trialend_marker)
      mrk_te = find(strcmpi(mrk_init.desc, opt.trialend_marker{m}));
      if ~isempty(mrk_te)
        ende(m) = max(mrk_te);
      end
    end
    ende = max(ende);
    mrk_init.desc = mrk_init.desc(start:ende);
    mrk_init.pos = mrk_init.pos(start:ende);
    
    TS = find(strcmpi(mrk_init.desc, opt.trialstart_marker));
    TE = [];
    for m = 1:length(opt.trialend_marker)
      TE = [TE find(strcmpi(mrk_init.desc, opt.trialend_marker{m}))];
    end
    TE = sort(TE);
    KP = find(strcmpi(mrk_init.desc, opt.response_marker));
    
    if length(TS) ~= length(TE)
        error('number of trials start and trial end markers must be equal!')
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % remove all trials with more than one or no keypress
    none = 0;
    more = 0;
    del_idx = [];
    for t = 1:length(TS)
        kp = find((TS(t)<KP) & (KP<TE(t)));
        if length(kp) > 1
            more = more+1;
            del_idx = [del_idx, TS(t):TE(t)];
        elseif isempty(kp)
            none = none+1;
            del_idx = [del_idx, TS(t):TE(t)];
        end
        if t~=length(TS)
            iv_kp = find((TE(t)<KP) & (KP<TS(t+1)));
            if ~isempty(iv_kp)
                del_idx = [del_idx, KP(iv_kp)];
            end
        end
    end
    
    if more>0
      disp([int2str(more) '/' int2str(length(TS)) ' trials removed (more than one keypress)'])
    end
    if none>0
      disp([int2str(none) '/' int2str(length(TS)) ' trials removed (no keypress)'])
    end
    
    if ~isempty(del_idx)
        keep_idx = setdiff(1:length(mrk_init.pos), del_idx);
        mrk_init.desc = mrk_init.desc(keep_idx);
        mrk_init.pos = mrk_init.pos(keep_idx);       
        %mrk_init = mrk_removeEvents(mrk_init, del_idx);
    end
    TS = find(strcmpi(mrk_init.desc, opt.trialstart_marker));
    KP = find(strcmpi(mrk_init.desc, opt.response_marker));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % associate keypresses with its respective target
    TAR = [];
    for kp = KP
        t = find(mrk_init.pos(kp)+precision(1) < mrk_init.pos & ...
                 mrk_init.pos(kp)+precision(2) > mrk_init.pos );
        t = setdiff(t, kp);  % remove keypress marker as a possible target
        if length(t) > 1
            warning(['There should only be one marker in the time interval around the keypress' ...
                     'Check if opt.time_bounds is too large.'])
            keyboard
        elseif length(t) == 1
            TAR = [TAR t];
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % store all target markers between the keypress-target and the trial start
    % & set new cls_nokp markers before the target markers
    mrk.pos = mrk_init.pos;
    mrk.desc = mrk_init.desc;
    mrk.fs = mrk_init.fs;    
    for tar = TAR
        %% create cls_kp marker
        r = rand;
        offset = r * cls_ival(1) + (1-r) * cls_ival(2); 
        mrk.pos(end+1) = round(mrk_init.pos(tar) + offset);
        mrk.desc{end+1} = opt.cls_kp_marker;
        
        %% create cls_nokp marker(s)
        ts = TS(max(find(TS<tar)));  % trial start of the respective target
        for marker = ts+1:tar-1
            if isTarget(marker)
                r = rand;
                offset = r * cls_ival(1) + (1-r) * cls_ival(2); 
                mrk.pos(end+1) = round(mrk_init.pos(marker) + offset);
                mrk.desc{end+1} = opt.cls_nokp_marker;
            end
        end
    end
    
    % sort markers chronologically
    [sord, sidx] = sort(mrk.pos);
    mrk.desc = mrk.desc(sidx);
    mrk.pos = mrk.pos(sidx);    
    %mrk = mrk_sortChronologically(mrk);
    
    
        
    function istarget = isTarget(idx)
        typ = mrk_init.desc(idx);
        if strcmpi(typ, opt.target_markers(1)) || ...
           strcmpi(typ, opt.target_markers(2)) || ...
           strcmpi(typ, opt.target_markers(3))
             istarget = 1;
        else
             istarget = 0;
        end
    end


end
   