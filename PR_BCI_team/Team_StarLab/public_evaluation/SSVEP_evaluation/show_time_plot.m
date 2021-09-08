%% raw EEG cap
sub = 2;

view_time = 0; %209
view_len = 4; %5
chan = {'C3','Cz','C4'};
scale = 30;

%% Train
epo = epo_train{sub};
epo = proc_selectChannels(epo, chan);

h1 = figure(1);
plot_each_channel_bbci(epo, [view_time view_time+view_len],'scale',scale,...
    'en_text',false,'title',[])
xticks(view_time:view_time+view_len);
xticklabels({0:view_len})

title('Train-standing')
h1.Position = [15 850 300 115];

%% Test
epo = epo_test{sub};
epo = proc_selectChannels(epo, chan);

h2 = figure(2);
plot_each_channel_bbci(epo, [view_time view_time+view_len],'scale',scale,...
    'en_text',false,'title',[])
xticks(view_time:view_time+view_len);
xticklabels({0:view_len})

title('Test-walking')
h2.Position = [15 550 300 115];
