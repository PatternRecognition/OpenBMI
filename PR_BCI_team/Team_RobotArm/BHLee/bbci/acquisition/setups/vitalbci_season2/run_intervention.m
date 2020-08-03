log_filename= [TODAY_DIR 'intervention' VP_CODE '.txt'];
fid= fopen(log_filename, 'w');
if fid==-1,
  error(sprintf('could not open log-file <%s> for writing', log_filename));
end
fprintf(fid, 'VP #%d - Code %s\n', VP_NUMBER, VP_CODE');
timestr=  datestr(now,'yyyy-mm-dd HH:MM:SS.FFF');
fprintf(fid, 'Start: %s\n',timestr);

switch(mod(VP_NUMBER, 3)),
 case 1,
  fprintf(fid, 'Intervention: PMR\n');
  system('start wmplayer');
  stimutil_waitForInput('msg_next','to start PMR.');
  %[snd, fs]= wavread([DATA_DIR 'studies/vitalbci_season2/Entspannung_PMR_22min_Yvonne_11025Hz.wav']);
  %wavplay(snd, fs);
  mp3file= [DATA_DIR 'studies/vitalbci_season2/Entspannung_PMR_22min_Yvonne.mp3'];
  system(sprintf('start wmplayer "%s"', mp3file));
  stimutil_waitForInput('msg_next','when finished.');
 case 2,
  fprintf(fid, 'Intervention: 2HAND training\n');
  stimutil_waitForInput('msg_next','to start 2HAND training.');
  run_2HAND_training;
  %stimutil_waitForInput('msg_next','when finished.');
 case 0,
  fprintf(fid, 'Intervention: control task\n');
  stimutil_waitForInput('msg_next','to start showing the PDF file.');
  pdffile= [DATA_DIR 'studies/vitalbci_season2/Kap71.pdf'];
  system(sprintf('start acrord32 /A "page=1&toolbar=0" "%s"', pdffile));
  pause(22*60);
  soundsc(sin((0:5000)/11025*pi*750), 22050); % beep!
  geometry= VP_SCREEN;
  geometry(1:2)= geometry(1:2) + 0.25* geometry(3:4);
  geometry(3:4)= 0.5*geometry(3:4);
  msg_fig= figure;
  desc= stimutil_readDescription('vitalbci_season2_control_intervention_time_over');
  stimutil_showDescription(desc, 'position',geometry, 'clf',1, 'waitfor','key');
  close(msg_fig);
end

timestr=  datestr(now,'yyyy-mm-dd HH:MM:SS.FFF');
fprintf(fid, 'End: %s\n',timestr);
fclose(fid);
