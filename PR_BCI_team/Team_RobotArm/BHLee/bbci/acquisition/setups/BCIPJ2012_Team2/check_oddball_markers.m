ISIstr= '100ms';
%ISIstr= '500ms';
%ISIstr= '1000ms';
%ISIstr= '1500ms';
file= ['Oddball_ISI' ISIstr 'VPlan'];
mrk= eegfile_readBVmarkers([TODAY_DIR file])
it= strmatch('S 20', mrk.desc);
int= strmatch('S 10', mrk.desc);
io= strmatch('S100', mrk.desc);
idx= sort([it; int; io]);
dd= diff(mrk.pos(idx))/mrk.fs*1000;
unique(dd)
ii= find(dd<2000);
fig_set(1);
hist(dd(ii))


file= 'calibration_CenterSpellerVPlan';
mrk= eegfile_readBVmarkers([TODAY_DIR file])
[toe, isr]= marker_mapping_SposRneg(mrk.desc);
istim= find(toe>=10 & toe<=50);
idx= isr(istim);
dd= diff(mrk.pos(idx))/mrk.fs*1000;
unique(dd)
ii= find(dd<2000);
fig_set(1);
hist(dd(ii))
