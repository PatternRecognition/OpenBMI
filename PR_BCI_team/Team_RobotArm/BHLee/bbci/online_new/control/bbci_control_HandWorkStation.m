function packet = bbci_control_HandWorkStation(cfy_out, event, opt)
%BBCI_CONTROL_HANDWORKSTATION - Generate control signal for Siemens
%
%Synopsis:
%  PACKET = bbci_control_HandWorkStation(CFY_OUT, EVENT, OPT)
%
%Arguments:
%  CFY_OUT - Output of the classifier
%  EVENT - 
%  OPT - Structure containing the field 'wld' with parameters
%
%Output:
% PACKET: Variable/value list in a CELL defining the control signal that
%     is to be sent via UDP to the SMA.

% 11-2011 MSK


persistent wld


%% initialize
if isempty(wld),
    wld = opt.wld;
    wld.speed.current = wld.speed.initial;
    packet = {wld.control_str, wld.speed.current};
    wld.y = [];
    wld.ni = 0;
    return
end


%% invoked by marker
if ~isempty(event.desc)
    if event.desc==wld.mrk.start_low
        wld.speed.current = wld.speed.calibration(1);
        wld.fig.induced = -1;
        packet = {};
    elseif event.desc==wld.mrk.start_high
        wld.speed.current = wld.speed.calibration(2);
        wld.fig.induced = 1;
        packet = {};
    elseif event.desc==wld.mrk.force_down
        packet = {wld.control_str, wld.speed.current};
    elseif event.desc==wld.mrk.force_up
        packet = {wld.control_str, wld.speed.current};
    end
    return
end


%% store fcy_out buffer
wld.ni = wld.ni+1;
wld.y = cat(1, wld.y, cfy_out);


%% apply workload detector and visualize progress
if strcmp(wld.strategy,'linfit')
    wld = online_detect_linfit(wld);
    wld = online_visualize_linfit(wld);
elseif strcmp(wld.strategy,'ranksum')
    wld = online_detect_ranksum(wld);
    wld = online_visualize_ranksum(wld);
end


%% send workload control packet
if wld.state.hi && wld.control
    wld.speed.current = wld.speed.current - wld.speed.delta;
    wld.speed.current = min(max(wld.speed.current,wld.speed.minmax(1)), wld.speed.minmax(2));
    packet = {wld.control_str, wld.speed.current};
    ppTrigger(wld.mrk.control_down);
elseif wld.state.lo && wld.control
    wld.speed.current = wld.speed.current + wld.speed.delta;
    wld.speed.current = min(max(wld.speed.current,wld.speed.minmax(1)), wld.speed.minmax(2));
    packet = {wld.control_str, wld.speed.current};
    ppTrigger(wld.mrk.control_up);
else
    packet = {};
end

