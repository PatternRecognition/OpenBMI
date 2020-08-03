word_list= {'WINKT','FJORD','LUXUS'; 
            'SPHINX','QUARZ','VODKA';
            'YACHT','GEBOT','MEMME'};

% Maximum acceptable deviation from fixation point in px
et_range = 180;   
use_eyetracker = 1;
          
% Reset random-generator seed to produce new random numbers
% Take cputime in ms as basis
rand('seed',cputime*1000)

bvr_sendcommand('checkimpedances');
fprintf('Prepare cap. Press <RETURN> when finished.\n');
pause

%system('cmd /C "d: & cd \svn\pyff\src & python FeedbackController.py -l debug -p FeedbackControllerPlugins  --additional-feedback-path=D:\svn\pyff_external_feedbacks" &')
system('cmd /C "d: & cd \svn\pyff-mit-eyetracker\src & python FeedbackController.py -l debug -p FeedbackControllerPlugins" &')
bvr_sendcommand('viewsignals');
pause(5)
send_xmlcmd_udp('init', general_port_fields.bvmachine, 12345);

%testrun
fprintf('Press <RET> to start the test-runs.\n');
pause


% Evaluate a command (such as setting variables)
c = input('Give command (press RETURN if no command): ');
if  numel(c)>0; eval(c); end;
fprintf('TESTRUN Matrix\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'P300Matrix', 'command', 'sendinit');
send_xmlcmd_udp('interaction-signal', 's:words',{'BEIN'});
send_xmlcmd_udp('interaction-signal', 's:datafilename','');
send_xmlcmd_udp('interaction-signal', 'i:screenWidth', 1280, 'i:screenHeight', 1024);
send_xmlcmd_udp('interaction-signal', 'i:et_fixate_center',0);
send_xmlcmd_udp('interaction-signal', 'i:et_range',et_range);
send_xmlcmd_udp('interaction-signal', 'i:use_eyetracker',use_eyetracker);
send_xmlcmd_udp('interaction-signal', 'i:fullscreen',1);
send_xmlcmd_udp('interaction-signal', 's:datafilename','');
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> to continue with tests measurement.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


% Evaluate a command (such as setting variables)
c = input('Give command (press RETURN if no command): ');
if  numel(c)>0; eval(c); end;
fprintf('TESTRUN Hex-o-Spell\n');
send_xmlcmd_udp('fc-signal', 's:TODAY_DIR','', 's:VP_CODE','', 's:BASENAME','');
send_xmlcmd_udp('interaction-signal', 's:_feedback', 'P300Hex');
send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
%fprintf('Start wrapper and press <RET>\n');
%pause
send_xmlcmd_udp('interaction-signal', 's:words',{'KARO'});
send_xmlcmd_udp('interaction-signal', 's:datafilename','');
send_xmlcmd_udp('interaction-signal', 'i:screenWidth', 1280, 'i:screenHeight', 1024);
send_xmlcmd_udp('interaction-signal', 'i:et_fixate_center',0);
send_xmlcmd_udp('interaction-signal', 'i:et_range',et_range);
send_xmlcmd_udp('interaction-signal', 'i:fullscreen',1);
send_xmlcmd_udp('interaction-signal', 'i:use_eyetracker',use_eyetracker);
send_xmlcmd_udp('interaction-signal', 's:datafilename','');
send_xmlcmd_udp('interaction-signal', 'command', 'play');
fprintf('Press <RET> to go for the real measurement.\n');
pause
send_xmlcmd_udp('interaction-signal', 'command', 'quit');


speller_tag= {'Matrix', 'Hex'};
speller_name= {'Matrix', 'Hex-o-Spell'};
fixate_name= {'Target', 'Center'};
satisfied= 0;
while ~satisfied,
  idx_block= randperm(12)-1;
  fixation= mod(idx_block, 2);
  speller_type= 1+mod(floor(idx_block/2), 2);
  i1= find(speller_type==1, 1, 'first');
  i2= find(speller_type==2, 1, 'first');
  satisfied= ~fixation(i1) && ~fixation(i2);
end
word_idx= 1+floor(idx_block/4);
for ib= 1:length(idx_block),
  % Evaluate a command (such as setting variables)
  c = input('Give command (press <ENTER> for no command): ');
  if  numel(c)>0; eval(c); end;
  % 
  fbname= ['visual_p300_' lower(speller_tag{speller_type(ib)})];
  fbname= [fbname sprintf('_%s', lower(fixate_name{1+fixation(ib)}))];
  fprintf('Block %02d: P300 %s', ib, speller_name{speller_type(ib)});
  fprintf(' with %s Fixation', fixate_name{1+fixation(ib)});
  fprintf(': %s\n', vec2str(word_list(word_idx(ib),:)));
  send_xmlcmd_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE, 's:BASENAME',fbname);
  send_xmlcmd_udp('interaction-signal', 's:_feedback', ['P300' speller_tag{speller_type(ib)}]);
  send_xmlcmd_udp('interaction-signal', 'command', 'sendinit');
  %fprintf('Start wrapper and press <RET>\n');
  %pause
  send_xmlcmd_udp('interaction-signal', 's:words',word_list(word_idx(ib),:));
  send_xmlcmd_udp('interaction-signal', 'i:screenWidth', 1280, 'i:screenHeight', 1024);
  send_xmlcmd_udp('interaction-signal', 'i:use_eyetracker',use_eyetracker);
  send_xmlcmd_udp('interaction-signal', 'i:et_fixate_center',fixation(ib));
  send_xmlcmd_udp('interaction-signal', 'i:et_range',et_range);
  send_xmlcmd_udp('interaction-signal', 'i:fullscreen',1);
  send_xmlcmd_udp('interaction-signal', 's:datafilename',[TODAY_DIR 'datafile.txt']);
  send_xmlcmd_udp('interaction-signal', 'command', 'play');
  pause(10)
  fprintf('Press <RET> to go for the next run.\n');
  pause
  send_xmlcmd_udp('interaction-signal', 'command', 'quit');
end

fprintf('Finished.\n');
