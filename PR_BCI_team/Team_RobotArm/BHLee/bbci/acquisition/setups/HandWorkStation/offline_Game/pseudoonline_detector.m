
function wld = pseudoonline_detector(wld)


if strcmp(wld.strategy,'linfit')
    wld = online_detect_linfit(wld);
    wld = online_visualize_linfit(wld);
elseif strcmp(wld.strategy,'ranksum')
    wld = online_detect_ranksum(wld);
    wld = online_visualize_ranksum(wld);
end

if wld.state.hi
    wld.state.hi = false;
elseif wld.state.lo
    wld.state.lo = false;
end
