
function wld = online_detect_ranksum(wld)


%% reset detector state and update speed history
wld.state.hi = false;
wld.state.lo = false;
wld.speed.history(wld.ni) = wld.speed.current;


%% force speed change ...
if wld.control && wld.ni > wld.force_ival
    ival_history = wld.speed.history(wld.ni-wld.force_ival+1:wld.ni);
    % (a) if no speed changes during force_ival
    if nnz(diff(ival_history)) == 0
        if wld.speed.current > mean(wld.speed.calibration)
            ppTrigger(wld.mrk.force_down)
            wld.fig.forced = -1;
            %wld.speed.current = wld.speed.calibration(1);
            wld.speed.current = wld.speed.current - wld.speed.delta;
            wld.speed.history(wld.ni) = wld.speed.current;
        else
            ppTrigger(wld.mrk.force_up)
            wld.fig.forced = 1;
            %wld.speed.current = wld.speed.calibration(2);
            wld.speed.current = wld.speed.current + wld.speed.delta;
            wld.speed.history(wld.ni) = wld.speed.current;
        end
    % (b) if largely extreme speeds during force_ival
    elseif mean(ival_history) > wld.speed.calibration(2)
        ppTrigger(wld.mrk.force_down)
        wld.fig.forced = -1;
        %wld.speed.current = wld.speed.calibration(1);
        wld.speed.current = wld.speed.current - wld.speed.delta;
        wld.speed.history(wld.ni) = wld.speed.current;
    elseif mean(ival_history) < wld.speed.calibration(1)
        ppTrigger(wld.mrk.force_up)
        wld.fig.forced = 1;
        %wld.speed.current = wld.speed.calibration(2);
        wld.speed.current = wld.speed.current + wld.speed.delta;
        wld.speed.history(wld.ni) = wld.speed.current;
    end
end


%% detect workload state using wilcoxon rank sum test
if wld.ni >= wld.wlen*2
    
    y0 = wld.y(wld.ni-wld.wlen*2+1:wld.ni-wld.wlen);
    y1 = wld.y(wld.ni-wld.wlen+1:wld.ni);
    p = ranksum(y0,y1);
    
    if mean(y0)<mean(y1)
        wld.state.n_hi = -log(p);
        wld.state.n_lo = 0;
    else
        wld.state.n_lo = -log(p);
        wld.state.n_hi = 0;
    end
    
    if wld.state.dead_time>0
        wld.state.dead_time = wld.state.dead_time-1;
    end
    
    if p < wld.alpha && wld.state.dead_time==0
        if mean(y0)<mean(y1)
            wld.state.hi = true;
            wld.state.dead_time = wld.dead_time;
        else
            wld.state.lo = true;
            wld.state.dead_time = wld.dead_time;
        end
    end
    
end




