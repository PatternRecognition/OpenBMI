function [sig t_long t_final] = createTapStimuli(opt,SOUND_DIR);

t_long=[0:1/(opt.fs):opt.act_time-(1/(opt.fs))];
t_sin = 0:1/opt.fs:(opt.act_time/(opt.act_time*opt.mod_freq))*0.2-(1/opt.fs);
t_zero =  0:1/opt.fs:(opt.act_time/(opt.act_time*opt.mod_freq))*0.8-(1/opt.fs);
t_final=[0:1/(opt.fs):opt.final_time-(1/(opt.fs))];

n_final=length(t_final);
n_sin = length(t_sin);
n_zero = length(t_zero);

win_ones = ones(1,n_sin);
win_zero = zeros(1,n_zero);

win_both = cat(2,win_ones,win_zero);
win_total = repmat(win_both,1,opt.act_time*opt.mod_freq);
npad = (opt.act_time*opt.fs)-length(win_total);
win_total_pad =cat(2,win_total,zeros(1,npad));

w_c = 2*pi*opt.carr_freq;

sig_sin = sin(w_c*t_long);

sig = sig_sin.*win_total_pad;

t_twitch_before = 0:1/(opt.fs):opt.twitch_start-(1/(opt.fs));
n_twitch_before =length(t_twitch_before);
win_twitch_before = ones(1,n_twitch_before);

t_twitch_on = 0:1/(opt.fs):opt.twitch_dura-(1/(opt.fs));
n_twitch_on =length(t_twitch_on);
win_twitch_on= opt.twitch_amp .*ones(1,n_twitch_on);

t_after = opt.final_time-(opt.twitch_start+opt.twitch_dura);

t_twitch_after = 0:1/(opt.fs):t_after-(1/(opt.fs));
n_twitch_after =length(t_twitch_after);
win_twitch_after = ones(1,n_twitch_after);

win_twitch_total = cat(2,win_twitch_before,win_twitch_on,win_twitch_after);

npad_twitch = n_final-length(win_twitch_total);
win_twitch_total_pad =cat(2,win_twitch_total,ones(1,npad_twitch));

if ~isempty(opt.final_time)

    sig=sig(1:n_final);
else
    t_final=[];
end

if ~isempty(opt.twitch_dura)
sig = sig.*win_twitch_total_pad;
end

wavwrite(sig,opt.fs,16,[SOUND_DIR 'cue_sssep\',opt.cuename,int2str(opt.mod_freq)])


end


