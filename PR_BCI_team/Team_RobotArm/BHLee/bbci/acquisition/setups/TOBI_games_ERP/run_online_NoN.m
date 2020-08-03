
input('DID YOU REALLY, REALLY DEBLOCK THE PARALLEL PORT?');

clc;
settings_bbci= {'bbci.start_marker', 251, ...
        'bbci.quit_marker', 254, ...
        'bbci.adaptation.running',0};
cmd_init= sprintf('dbstop if error ; VP_CODE= ''%s''; TODAY_DIR= ''%s''; set_general_port_fields(''localhost''); general_port_fields.feedback_receiver = ''tobi_c'';  global acquire_func; acquire_func=@%s;', VP_CODE, TODAY_DIR, func2str(acquire_func));
% cmd_init= sprintf('dbstop if error ; VP_CODE= ''%s''; TODAY_DIR= ''%s''; set_general_port_fields(''localhost''); general_port_fields.feedback_receiver = ''matlab'';  global acquire_func; acquire_func=@acquire_sigserv;', VP_CODE, TODAY_DIR);
bbci_cfy= [TODAY_DIR 'bbci_classifier.mat'];
cmd_bbci= ['dbstop if error; bbci_bet_apply(''' bbci_cfy ''')'];
% system(['matlab -nosplash -nojvm -r "' cmd_init cmd_bbci '; exit &']); 
eval([cmd_init cmd_bbci]);

disp('Classifier started.... ');

pause(4);    

[dum fname] = fileparts(bbci.online_file);
% signalServer_startrecoding(fname)
disp('recording started.... Please start application by hand.');

inp = '';
while ~strcmp(inp, 'stop')
inp = input('type ''stop'' to quit the online game: ', 's');
end
ppTrigger(254)


disp('EEG data acquisition was stopped')

