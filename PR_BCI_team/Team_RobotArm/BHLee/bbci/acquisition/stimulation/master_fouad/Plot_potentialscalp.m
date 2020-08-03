
file = 'D:\data\bbciRaw\VPiz_08_04_15\fingertip_leftVPiz';
[cnt,mrk]= eegfile_loadBV(file,'fs',100);

mnt= getElectrodePositions(cnt.clab)

text(mnt.x, mnt.y, mnt.clab)
axis([-1 1 -1 1])
% scalpPlot(mnt, cnt.x(2000,:))
for i= 1:200; 
  scalpPlot(mnt, cnt.x(88284+10*i,:)); 
  drawnow; 
  pause(0.05); 
end

miscDef= {102;'start'};
mrk_misc= mrk_defineClasses(mrk, miscDef);

epo = cntToEpo(cnt,mrk_misc,[0 40000])
