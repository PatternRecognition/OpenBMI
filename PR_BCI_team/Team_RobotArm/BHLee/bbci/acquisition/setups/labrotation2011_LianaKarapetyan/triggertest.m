%session_name= 'labrotation2011_LianaKarapetyan';
%addpath([BCI_DIR 'acquisition/setups/' session_name]);

VP_CODE= 'Trigger';
global TODAY_DIR
acq_makeDataFolder;

start_signalserver('server_config_visualERP_gSAHARA.xml'); pause(1);
opt_rec= struct('quit_marker',255, 'position',[-2555 920 155 80]);
if 0,
  cmd= [BCI_DIR 'online\communication\signalserver\Scope\TOBI_RemoteScope.exe &'];
  system(cmd);
end


signalServer_startrecoding(['triggertest_' VP_CODE], opt_rec); pause(10);
waitForSync;
k= 0;
for i= 1:10*255,
  k= 1 + mod(k,254);
  ppTrigger(k);
  waitForSync(100);
end
pause(1);
ppTrigger(255);



file= bvr_startrecording(['triggertestbv_' VP_CODE], 'impedances',0); 
for i= 1:50*254,
  k= 1 + mod(i,254);
  ppTrigger(k);
  pause(0.05);
end
pause(1);
ppTrigger(255);
bvr_sendcommand('stoprecording');
mrk= eegfile_readBVmarkers(file)
dd= diff(mrk.pos/mrk.fs*1000);
plot(dd(2:end-1));



file= signalServer_startrecoding(['triggertestgtec_' VP_CODE], opt_rec); pause(10);
for i= 1:50*254,
  k= 1 + mod(i,254);
  ppTrigger(k);
  pause(0.05);
end
pause(1);
ppTrigger(255);
mrk= eegfile_readBVmarkers(file)
dd= diff(mrk.pos/mrk.fs*1000);
plot(dd(2:end-1));



opt_rec= struct('quit_marker',255, 'position',[-2555 920 155 80]);
file_gt= signalServer_startrecoding(['triggertestgtec_' VP_CODE], opt_rec); pause(10);
file_bv= bvr_startrecording(['triggertestbv_' VP_CODE], 'impedances',0); 
waitForSync;
for i= 1:20*254,
  k= 1 + mod(i,254);
  ppTrigger(k);
  waitForSync(50);
end
pause(1);
ppTrigger(255);
bvr_sendcommand('stoprecording');
