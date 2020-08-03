function cnt = filter_bandpass(cnt, varargin)
% Performs high- and low-pass filtering.
%
% sven.daehne@tu-berlin.de

opt= propertylist2struct(varargin{:});
% default filter options
opt = set_defaults(opt, ...
    'do_highpass_filter', 1, ...
    'hp_transition_band', [0.1, 0.4], ... %
    'hp_attenuation', [3, 50], ... % max attenutation in pass band, min attenuation in stop band
    'do_lowpass_filter', 1, ...
    'lp_transition_band', [40, 49], ... %
    'lp_attenuation', [3, 50], ... % max attenutation in pass band, min attenuation in stop band
    'use_causal_filter', 0, ... % causal means use proc_filt, non-causal means use proc_filtfilt
    'verbose', 1 ...
    );


if opt.do_lowpass_filter
    if opt.verbose
        display('filtering cnt lowpass')
    end
    % design low-pass filter
    Wps = opt.lp_transition_band/cnt.fs*2;
    [n, Ws]= cheb2ord(Wps(1), Wps(2), opt.lp_attenuation(1), opt.lp_attenuation(2));
    [filt.b, filt.a]= cheby2(n, opt.lp_attenuation(2), Ws);
    if opt.use_causal_filter
        cnt = proc_filt(cnt, filt.b, filt.a);
    else
        cnt = proc_filtfilt(cnt, filt.b, filt.a);
    end
end

if opt.do_highpass_filter
    if opt.verbose
        display('filtering cnt highpass')
    end
    Wps = opt.hp_transition_band(1:2)/cnt.fs*2;
    [n, Wn] = buttord(Wps(1), Wps(2), opt.hp_attenuation(1), opt.hp_attenuation(2));
    [filt.b, filt.a]= butter(n, Wn, 'high');
    if opt.use_causal_filter
        cnt = proc_filt(cnt, filt.b, filt.a);
    else
        cnt = proc_filtfilt(cnt, filt.b, filt.a);
    end
end