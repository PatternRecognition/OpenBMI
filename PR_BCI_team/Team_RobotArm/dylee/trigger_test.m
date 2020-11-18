VP_CODE= 'Trigger';
acq_makeDataFolder;

nRounds= 100;
bvr_startrecording('test', 'impedances',0);
pause(1);
for r= 1:nRounds,
  fprintf('round %d\n', r);
  for t= 1:255,
    ppTrigger(t);
    pause(0.05);
  end
end
pause(1);
bvr_sendcommand('stoprecording');

file= [TODAY_DIR 'test'];
mrk= eegfile_readBVmarkers(file);
for t= 1:255,
  trig= sprintf('S%3d', t);
  oc(t)= length(strmatch(trig, mrk.desc));
end
missed= find(oc<nRounds);
plot(oc, '.');
hold on
plot(missed, oc(missed),'r.');
hold off
fprintf('missed markers: %d\n', 255*nRounds-sum(oc));
