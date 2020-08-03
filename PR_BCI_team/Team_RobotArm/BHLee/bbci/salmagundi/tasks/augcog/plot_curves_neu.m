di = '/home/schlauch/dornhege/augcog_season4/';


d = dir(di);

ind = [];
for i = 1:length(d)
  if length(d(i).name)>4 & strcmp(d(i).name(end-3:end),'.out')
    ind = [ind,i];
  end
end



d = d(ind);


for fi = 1:length(d);

  for ta = 1:2
S = load([di d(fi).name(1:end-4) '.out'],'-mat');
out = S.out;

na = d(fi).name;
na = na(16:end-4);
ye = str2num(na(1:4));
mo = str2num(na(5:6));
da = str2num(na(7:8));
ho = str2num(na(10:11));
mi = str2num(na(12:13));
se = str2num(na(14:15));

switch da
 case 17
  nam = 'ab';
 case 18
  nam = 'ar';
 case 19
  nam = 'th';
 case 24
  nam = 'um';
 case 12
  nam = 'ts';
end


switch ta
 case 1
  ind = find(out.toe==1);
  ind2 = find(out.toe==3);
  S = load([di nam '_calc_classifier.cl'],'-mat');
  classifier = getfromdouble(S.classifier);
  tresh = classifier.mapping;
  
 case 2
  ind = find(out.toe==2);
  ind2 = find(out.toe==4);
  S = load([di nam '_audio_classifier.cl'],'-mat');
  classifier = getfromdouble(S.classifier);
  tresh = classifier.mapping;
end  
  ind3 = find(out.toe==6);

in = [];

for i = 1:length(ind)
  ccc = ind-ind(i);
  ccc = ccc(ccc>0);
  cc = cat(2,ind2-ind(i),ind3-ind(i));
  cc = cc(cc>0);
  if ~isempty(ccc) 
    if isempty(cc)
      in = cat(2,in,[ind(i);length(out.toe)]);
    else
      if min(ccc)>min(cc)
        in = cat(2,in,[ind(i);min(cc)+ind(i)]);
      end
    end
  else
    if isempty(cc)
      in = cat(2,in,[ind(i);length(out.toe)]);
    else
      in = cat(2,in,[ind(i);min(cc)+ind(i)]);
    end
  end
end

for i = 1:length(ind2)
  ccc = ind2-ind2(i);
  ccc = ccc(ccc>0);
  cc = cat(2,ind-ind2(i),ind3-ind2(i));
  cc = cc(cc>0);
  if ~isempty(ccc) 
    if isempty(cc)
      in = cat(2,in,[ind2(i);length(out.toe)]);
    else
      if min(ccc)>min(cc)
        in = cat(2,in,[ind2(i);min(cc)+ind2(i)]);
      end
    end
  else
    if isempty(cc)
      in = cat(2,in,[ind2(i);length(out.toe)]);
    else
      in = cat(2,in,[ind2(i);min(cc)+ind2(i)]);
    end
  end
end

if isempty(in)
  continue;
end


[in(1,:),ind] = sort(in(1,:));
in(2,:) = in(2,ind);

xx = cell(1,size(in,2));
out.y = zeros(2,size(in,2));
for i = 1:size(in,2);
  xx{i} = out.x(:,out.pos(in(1,i)):out.pos(in(2,i)));
  out.y((out.toe(in(1,i))==ta+2)+1,i) = 1;
end

out.x = xx;
out.className = {'high','base'};

clear mr
step = 0;
data = [];
for i = 1:length(out.x)
  data = cat(2,data,[step+1:step+size(out.x{i},2);out.x{i}([ta,ta+2],:)]);
  mr(i) = step+1;
  step = step+size(out.x{i},2);
end
  
clf;
subplot(2,1,1);
hold on;
plot(data(1,:),data(2,:),'k');

yy = get(gca,'YLim');
clf;
subplot(2,1,1);
hold on;


for i = 1:length(mr)
  l = line([1 1]*mr(i),yy);
  set(l,'Linewidth',3);
  switch find(out.y(:,i))
   case 2
    set(l,'Color',[0 0 1]);
   case 1
    set(l,'Color',[1 0 0]);
  end
end

plot(data(1,:),data(2,:),'k');

l = line([1,step],[tresh(1),tresh(1)]);
set(l,'Color',[0 0 0],'LineStyle','--');
l = line([1,step],[tresh(2),tresh(2)]);
set(l,'Color',[0 0 0],'LineStyle','--');
l = line([1,step],[0,0]);
set(l,'Color',[0 0 0],'LineStyle','-');

tim = size(data,2)/out.fs/60;

set(gca,'XTick',(1:floor(tim))*60*out.fs);
set(gca,'XTickLabel',1:floor(tim));
set(gca,'XLim',[1,step]);
title('workload indicator (classifier output)','fontsize',16);

    
subplot(2,1,2);
hold on;

plot(data(1,:),data(3,:),'k','linewidth',2);

set(gca,'XTick',mr);
set(gca,'XTickLabel',out.className([1,2]*out.y));
set(gca,'XLim',[1,step]);
title('mitigation request','fontsize',16);


p = out.pos(in(1,1))/out.fs/60/60/24;

n = datenum(ye,mo,da,ho,mi,se)+p;

[ye,mo,da,ho,mi,se] = datevec(n);
se = round(se);

set(gcf,'PaperOrientation','landscape');
switch ta
 case 1
  addTitle(sprintf('%i/%i/%i %i:%i:%i %s %s',ye,mo,da,ho,mi,se,nam,'calc'));
  saveFigure(sprintf('%straces_neu_%i%i%i%i%i%i_calc',di,ye,mo,da,ho,mi,se),'maxAspect');
 case 2
  addTitle(sprintf('%i/%i/%i %i:%i:%i %s %s',ye,mo,da,ho,mi,se,nam,'audio'));
  saveFigure(sprintf('%straces_neu_%i%i%i%i%i%i_audio',di,ye,mo,da,ho,mi,se),'maxAspect');
end

end

end


