global IMPORT_DIR
addpath([IMPORT_DIR 'ica/cardoso/']);

file= 'Thorsten_03_04_03/react1_5sThorsten';
[cnt, mrk, mnt]= loadProcessedEEG(file);

displayChannels= find(~isnan(mnt.x));
cnt= proc_selectChannels(cnt, displayChannels);


ival= mrk.trg.pos([1 end]);
C= cov(cnt.x(ival(1):ival(end),:));
ev= eig(C, 'nobalance');
plot(log(ev));

erp= makeEpochs(cnt, mrk.trg, [0 1500]);
erp= proc_average(erp);
mnt= restrictDisplayChannels(mnt, erp_sth);

nSources= 10;

x_left= erp.x(:,:,1);
Al= jadeR(x_left, nSources);
Sl= Al*x_left;
erp_sth= copyStruct(erp, 't','y','className');
erp_sth.x= reshape(Sl, [size(Sl) 1]);
erp_sth.y= 1;
erp_sth.className= {'left'};


opt= struct('resolution',24, 'shading','flat', 'scalePos','none');
figure(1);
clf;
for n= 1:nSources,
  suplot(nSources, n, [0.1 0.1 0]);
  plotScalpPattern(mnt, erp_sth.x(n,:), opt);
  hl= ylabel(sprintf('N=%d', n));
  set(hl, 'visible','on');
  suplot(nSources, n, [0.005 0.25 0.25], [0.01 0.01 0.01]);
  plot(Al(n,:));
  set(gca, 'xLim',[1 size(Al,2)], 'xTick',[], 'yTick',[]);
end


x_right= erp.x(:,:,2);
Ar= jadeR(x_right, nSources);
Sr= Ar*x_right;
erp_sth= copyStruct(erp, 't','y','className');
erp_sth.x= reshape(Sr, [size(Sr) 1]);
erp_sth.y= 1;
erp_sth.className= {'right'};

figure(2);
clf;
for n= 1:nSources,
  suplot(nSources, n, [0.1 0.1 0]);
  plotScalpPattern(mnt, erp_sth.x(n,:), opt);
  hl= ylabel(sprintf('N=%d', n));
  set(hl, 'visible','on');
  suplot(nSources, n, [0.005 0.25 0.25], [0.01 0.01 0.01]);
  plot(Ar(n,:));
  set(gca, 'xLim',[1 size(Ar,2)], 'xTick',[], 'yTick',[]);
end
