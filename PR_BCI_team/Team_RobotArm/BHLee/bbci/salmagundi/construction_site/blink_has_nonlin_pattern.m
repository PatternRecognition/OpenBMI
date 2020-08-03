cd([BCI_DIR 'studies/season2']);

file= 'VPcm_06_02_21/arteVPcm';

[cnt, mrk, mnt]= eegfile_loadMatlab(file);
cnt= rmfield(cnt, 'title');

ct = proc_selectChannels(cnt,'EOGv');
ct = proc_subtractMovingAverage(ct,2000);
ct = proc_movingAverage(ct,200);
ct = proc_rectifyChannels(ct);

%% blinking period
cB= getClassIndices(mrk, 'blinking');
iB= max(find(mrk.y(cB,:)));    %% max should not be neccessary
iS= min(find(mrk.pos>mrk.pos(iB) & mrk.y(1,:)));
x= ct.x(mrk.pos(iB):mrk.pos(iS));

%% little blinking period
cEO= getClassIndices(mrk, 'eyes open');
iEO= find(mrk.y(cEO,:));
iS= min(find(mrk.pos>mrk.pos(iEO) & mrk.y(1,:)));
nx= ct.x(mrk.pos(iEO):mrk.pos(iS));

%% determine 'blink threshold'
f = inline('mean(nx>t)+mean(x<t)','t','x','nx');
%tr = fminbnd(f,0,max([x; nx]),optimset,x,nx);
%tr = fminsearch(f,35,optimset,x,nx);
tr = 2*min_by_linesearch(f,0,max([x;nx]),x,nx);

subplot(211); 
plot(x); title('blinking periods');
line(xlim, [tr tr], 'color','r');
subplot(212); 
plot(nx); 
line(xlim, [tr tr], 'color','r');

yAboveThresh= [ct.x>tr];
iBlinkStart= 1 + find(diff(yAboveThresh)==1);
iBlink= zeros(1, length(iBlinkStart));
for bb= 1:length(iBlink),
  iBlinkEnd= iBlinkStart(bb) + ...
      min(find(diff(yAboveThresh(iBlinkStart(bb):end))==-1)) - 1;
  [mm,mi]= max(ct.x([iBlinkStart(bb) iBlinkEnd]));
  iBlink(bb)= iBlinkStart(bb) + mi -1;
end
mk= struct('fs',mrk.fs, 'pos',iBlink, 'className',{{'blink'}}, ...
           'y',ones(1,length(iBlink)));

epo= makeEpochs(cnt, mk, disp_ival);
epo= proc_baseline(epo, [-400 -200]);
erp= proc_average(epo);

clf;
plotChannel(erp, {'Fz','Cz','Pz','Oz'});
