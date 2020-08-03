
function wld = online_detect_linfit(wld)

if wld.ni<wld.wlen
    return
else
    wld.y_lp(wld.ni-wld.wlen+1) = mean(wld.y(wld.ni-wld.wlen+1:wld.ni));
end

if wld.ni<wld.wlen*2
    return
else    
    y_lp = wld.y_lp(wld.ni-wld.wlen*2+2:wld.ni-wld.wlen+1);
    
    % compute linear trend
    p = polyfit(1:wld.wlen,zscore(y_lp),1);
    if p(1) > wld.thresh
        wld.state.n_hi = wld.state.n_hi+1;
        wld.state.n_lo = 0;
    elseif p(1) < -wld.thresh
        wld.state.n_lo = wld.state.n_lo+1;
        wld.state.n_hi = 0;
    else
        wld.state.n_hi = 0;
        wld.state.n_lo = 0;
    end
    
    % check for condition
    if wld.state.n_hi==wld.wlen
        wld.state.hi = true;
        wld.state.n_hi = 0;
    elseif wld.state.n_lo==wld.wlen
        wld.state.lo = true;
        wld.state.n_lo = 0;
    end
    
end

