vp= 0;
addpath([BCI_DIR 'acquisition/setups/labrotation2010_DmitryZarubin'])
subdir_list= get_session_list('leitstand_season2');
%subdir_list= get_session_list('season10');

%% -- start copy from here and paste repeatedly
close all
vp= vp+1;
subdir= subdir_list{vp}
sbj= subdir(1:find(subdir=='_',1,'first')-1);
bbci= [];
bbci.train_file= strcat(subdir, '/relax', sbj);

global TODAY_DIR
TODAY_DIR= '/tmp/';
bbci.setup= 'smr_extractor';
bbci.func_mrk= 'durchrauschen';
bbci.save_name= strcat(TODAY_DIR, 'bbci_smr_extractor');
bbci.feedback= '';
bbci_bet_prepare
mrk_orig= mrk;
bbci_bet_analyze
bbci_bet_finish
%% -- stop for copy and paste


%% -- dry feedback run (classifier output only)
%set_general_port_fields({'gesualdo','gesualdo'});
%bbci_bet_apply_offline(Cnt, mrk_orig, 'setup_list',bbci.save_name);


%% -- feedback run with simulated feedback:
addpath([BBCI_DIR 'simulation/compatibility']);
close all
feedback_opt= struct('type', 'feedback_smr_bar');
bbci_bet_apply_offline(Cnt, mrk_orig, ...
                       'setup_list',bbci.save_name, ...
                       'modifications',{'bbci.fb_machine', '', ...
                                        'feedback_opt',feedback_opt});

%% -- the same in real-time:
bbci_bet_apply_offline(Cnt, mrk_orig, ...
                       'realtime', 1, ...
                       'setup_list',bbci.save_name, ...
                       'modifications',{'bbci.fb_machine', '', ...
                                        'feedback_opt',feedback_opt});

%% -- with PyFF Feedback
general_port_fields.feedback_receiver= 'pyff';
bbci_bet_apply_offline(Cnt, mrk_orig, ...
                      'realtime', 1, ...
                      'setup_list',bbci.save_name, ...
                      'modifications',{'bbci.fb_machine', '127.0.0.1', ...
                                       'bbci.fb_port', 12345});

%% In another Matlab:
%% this does not work so far:
system(['cd ~/svn/pyff/src; python FeedbackController.py --nogui -l debug --additional-feedback-path=/home/blanker/svn/ida/public/bbci/python/pyff/src/Feedbacks &']);

system(['cmd /C "D: & cd \svn\pyff\src & python FeedbackController.py --port=0x' dec2hex(IO_ADDR) ' --nogui -l debug -p brainvisionrecorderplugin" &']);
pause(8)

send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'SMR_NeuroFeedback','command','sendinit'); 
send_xmlcmd_udp('interaction-signal', 'command', 'play');
