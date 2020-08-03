eeg_file = 'I:\EEG_Import\AugCog\season_4\17.08.2004\ab_ref_1';
file = 'I:\EEG_Import\AugCog\season_4\17.08.2004\ab_ref_1-0002.avi';
log_file = 'I:\EEG_Import\AugCog\season_4\17.08.2004\classifier_log_20040817T125707.out';
log_file2 = 'I:\EEG_Import\AugCog\season_4\17.08.2004\classifier_log_20040817T125707.log';

clas = {'I:\EEG_Import\AugCog\season_4\17.08.2004\ab_calc_classifier.cl','I:\EEG_Import\AugCog\season_4\17.08.2004\ab_audio_classifier.cl'};

timeshift = 70;

fid = fopen([eeg_file '.videoconfig']);
s = '';

while isempty(strmatch('File2',s))
    s = fgets(fid);
end

fclose(fid);

c = strfind(s,',');
s = s(c(1)+1:c(2)-1);
yed = str2num(s(1:4));
mod = str2num(s(5:6));
dad = str2num(s(7:8));
hod = str2num(s(9:10));
mid = str2num(s(11:12));
sed = str2num(s(13:14));
nn = datenum(yed,mod,dad,hod,mid,sed)*24*60*60+timeshift;

thresh = zeros(length(clas),2);

for i = 1:length(clas)
    S = load(clas{i},'-mat');
    classifier = getfromdouble(S.classifier);
    thresh(i,:) = classifier.mapping;
end
    

fid = fopen(log_file2,'r');
s = fgets(fid);
while isempty(strmatch('Start',s));
    s = fgets(fid);
end
fclose(fid);

s = s(18:end);
[ye,mo,da,ho,mi,se] = datevec(s);
n = datenum(ye,mo,da,ho,mi,se)*24*60*60;

S = load(log_file,'-mat');
out = S.out;



inf = aviinfo(file);

file2 = 'I:\EEG_Import\AugCog\season_4\17.08.2004\ab_ref_1-0001_modi.avi';
mo = avifile(file2,'fps',inf.FramesPerSecond,'Compression','DIV4');

idd = 100;

printprogress(1);

set(gcf,'Position',[-300 1000 200 100]);
pos = [inf.Height,inf.Width];

bal = round([pos*0.95,pos*0.99]);

po_old = 0;


sc = 1;

hh = figure;
set(hh,'MenuBar','none','Position',[100 100 sc*pos([2,1])]);
set(gca,'Position',[0,0,1,1]);
axis off

clear S

pospi = get(hh,'Position');
siz = 0.8;
pospi = [pospi(1)+pospi(3)+30,pospi(2),round(siz*pos([2,1])*sc)];

info = visualize_classifier({'calc','audio'},thresh,pospi);
for i = 1:ceil(inf.NumFrames/idd)
%for i = 1:20
    
    mov = aviread(file,(i-1)*idd+1:min(i*idd,inf.NumFrames-2));
    
 %   dat = repmat(uint8(zeros([round(sc*pos),3])),[1,1,1,length(mov)]);
    
    if sc~=1
    set(hh,'Visible','on');
    figure(hh);
    for j = 1:length(mov)
  %      printprogress((i-1)*idd+j,inf.NumFrames);    
        image(mov(j).cdata); colormap(mov(j).colormap);
        set(gca,'XTick',[]);
        set(gca,'YTick',[]);
        data = getframe(hh);
        mov(j).cdata = data.cdata;
    end
    end 
    
    set(hh,'Visible','off');
%    set(info,'Visible','off');
    figure(info);
    for j = 1:length(mov);
        printprogress((i-1)*idd+j,inf.NumFrames);    
        tip = (i*idd+j)/inf.FramesPerSecond+nn-n;
        tip = tip*out.fs;
        tip = round(tip);
        if tip>=1 & tip<= size(out.x,2)
            ind = find(out.pos<=tip);
            ind = sort(ind);
            ind = ind(max(1,length(ind)-3):end);
            mrk = out.toe(ind);
             
            
            visualize_classifier(out.x(:,tip),mrk);
            pi = getframe(info);
            pi = pi.cdata;
            dat = mov(j).cdata;
    switch 2
        case 1
            dat = cat(1,dat,uint8(0*ones(size(dat))));
            dat = cat(2,dat,cat(1,uint8(0*ones(size(pi))),pi));
        case 2
            if size(dat,1)<size(pi,1);
                dat = cat(1,dat,uint8(0*ones(size(pi,1)-size(dat,1),size(dat,2),size(dat,3))));
            end
            if size(dat,1)>size(pi,1);
                pi = cat(1,pi,uint8(0*ones(size(dat,1)-size(pi,1),size(pi,2),size(pi,3))));
            end
            
            dat = cat(2,dat,pi);
    end
        end
        
        mo= addframe(mo,dat);
            
    end
%    set(info,'Visible','off');

    
    
%     for j = 1:size(bal,1) 
%         po_old = po_old+randn;
%         if po_old>0
%             mov.cdata(bal(j,1):bal(j,3),bal(j,2):bal(j,4),:) = repmat(permute([255,0,0],[1 3 2]),[bal(j,3)-bal(j,1)+1,bal(j,4)-bal(j,2)+1,1]);
%         else
%             mov.cdata(bal(j,1):bal(j,3),bal(j,2):bal(j,4),:) = repmat(permute([0,255,0],[1 3 2]),[bal(j,3)-bal(j,1)+1,bal(j,4)-bal(j,2)+1,1]);
%         end             
%     end
    
end

mo = close(mo);
    
printprogress;

close all;