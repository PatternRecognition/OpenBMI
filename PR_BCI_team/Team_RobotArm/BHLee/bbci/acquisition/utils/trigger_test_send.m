%VP_CODE= 'Trigger';
%acq_makeDataFolder;

nRounds= 100;
%bvr_startrecording('test', 'impedances',0);
%pause(1);
for r= 1:nRounds,
  fprintf('round %d\n', r);
  for t= 1:255,
    ppTrigger(t);
    pause(0.05);
  end
end
%pause(1);
%bvr_sendcommand('stoprecording');
