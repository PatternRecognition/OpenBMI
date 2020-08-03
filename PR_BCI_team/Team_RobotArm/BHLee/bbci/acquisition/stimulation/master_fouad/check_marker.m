file = 'D:\data\bbciRaw\test_master_fouad_150408';
[cnt,mrk]= eegfile_loadBV(file,'clab',[1:20]);

for i=5:53
  
  act = mrk.pos(i)-mrk.pos(i-1)
end

mrk_pos = mrk.pos(20)-mrk.pos(3) 
mrk_pos_1 = mrk.pos(37)-mrk.pos(20) 
mrk_pos_2 = mrk.pos(end)-mrk.pos(37) 