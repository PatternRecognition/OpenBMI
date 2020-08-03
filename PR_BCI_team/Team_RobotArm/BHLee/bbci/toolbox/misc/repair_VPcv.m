file_list = {'VPcv_06_03_30/imag_1drfbVPcv','VPcv_06_03_30/imag_fbspeller_lettersVPcv','VPcv_06_03_30/imag_fbspeller_wordsVPcv'};

log_info = load_logfile('VPcv_06_03_30/imag_VPcv_setup_001_log');
log_info = map_logmarker(log_info, 'stimulus');
log_fb = load_feedback('VPcv_06_03_30/log');
for i = 1:length(log_fb)
  ind = find(log_fb(i).update.counter==0);
  if ~isempty(ind),log_fb(i).update.counter(ind) = log_fb(i).update.counter(ind(end)+1)-1;end
  ind = find(isnan(log_fb(i).update.pos));
  if ~isempty(ind),log_fb(i).update.pos(ind) = log_fb(i).update.pos(ind(end)+1)-4;end
  ind = find(isnan(log_fb(i).update.lognumber));
  if ~isempty(ind),log_fb(i).update.lognumber(ind) = log_fb(i).update.lognumber(ind(end)+1);end
end

log_inf = log_info;
for i = length(log_fb):-1:1
  mrk = log_fb(i).mrk;
  mrk.pos = mrk.pos+6;% delay!!!!
  ind = find(mrk.pos<log_inf.mrk.pos(1));
  log_inf.mrk.pos = [mrk.pos(ind),log_inf.mrk.pos];
  log_inf.mrk.toe = [mrk.toe(ind),log_inf.mrk.toe];
  [log_inf.mrk.pos,ind] = sort(log_inf.mrk.pos);
  log_inf.mrk.toe = log_inf.mrk.toe(ind);
end

log_info_back = log_info;
log_info = log_inf;

log_info.time(6)=log_info.time(6)+29.5;
for i = [6,5]
  while log_info.time(i)>=60
    log_info.time(i-1) = log_info.time(i-1) +1;
    log_info.time(i) = log_info.time(i) -60;
  end
end

for fi = 1:length(file_list)
  [Cnt,Mrk] = eegfile_loadBV(file_list{fi},'fs',1000,'prec',1);
  Mrk = Mrk(1);
  dat  =Mrk.time;
  ho = str2num(dat(9:10));
  mi = str2num(dat(11:12));
  se = str2num(dat(13:14));
  ms = str2num(dat(15:17));
  tidif = log_info.time(4:6)-[ho mi se+0.001*ms];
  tidif = tidif*[60*60,60,1]'*100;
  posis = round(log_info.mrk.pos+tidif);
  ind = find(posis>0 & posis<size(Cnt.x,1)/Mrk(1).fs*log_info.fs);
  for j = 1:length(ind)
    Mrk(end+1).type = 'Stimulus';
    Mrk(end).desc = sprintf('S% 3d',log_info.mrk.toe(ind(j)));
    Mrk(end).pos = posis(ind(j))*Mrk(1).fs/log_info.fs;
    Mrk(end).length = 1;
    Mrk(end).chan = 0;
    Mrk(end).time = '';
  end
  Cnt.title = [EEG_RAW_DIR Cnt.title];
  eegfile_writeBV(Cnt,Mrk);
end

save([EEG_RAW_DIR 'VPcv_06_03_30/reconstructed_log_files'],'log_info','log_fb');

global problem_marker_bit
problem_marker_bit=1;

prepare_data_bbci_bet('VPcv_06_03_30','log_info',log_info,'log_fb',log_fb,'file',{'imag_fbhexa*','imag_fbspeller_letters2VPcv'});

problem_marker_bit = 0;
prepare_data_bbci_bet('VPcv_06_03_30','log_info',log_info,'log_fb',log_fb,'file',{'arteVPcv','imag_lettVPcv','imag_fbspeller_lettersVPcv','imag_fbspeller_wordsVPcv','imag_1drfbVPcv'});
