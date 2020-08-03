function add_classifier_to_movie(eeg_file,timeshift,clas,idd,names, messages,moviefile);
%ADD_CLASSIFIER_TO_MOVIE ADDS THE FRAUNHOFER CLASSIFIER TO THE MOVIE
%
% usage:
%    add_classifier_to_movie(eeg_file,<timeshift,clas,idd,names, messages,timearray,moviefile>);
%
% input:
%    eeg_file      the name of an eeg_file 
%    timeshift     timeshift in sec between classifier log file and video [0]
%    clas          the name of classifier files to load 
%                  [{'ab_calc_classifier.cl','ab_audio_classifier.cl'}]
%    idd           Number of frames of the movie to load
%    names         Name of the classifiers [{'calc','audio'}]
%    messages      flag, if messages about block numbers should appear [true]
%    moviefile      the name of the moviefile
%
% Guido Dornhege, 07/09/2004

if ~exist('timeshift','var') | isempty(timeshift)
  timeshift = 0;
end

if ~exist('clas','var') | isempty(clas)
  clas = {'ab_calc_classifier.cl','ab_audio_classifier.cl'};
end

if ~exist('names','var') | isempty(names)
  names = {'calc','audio'};
end


if ~exist('idd','var') | isempty(idd)
  idd = 100;
end

if ~exist('messages','var') | isempty(messages)
  messages = 1;
end

if ~exist('moviefil','var')
  moviefil = '';
end


%eeg_file = 'I:\EEG_Import\AugCog\season_4\17.08.2004\ab_ref_1';


fid = fopen([eeg_file '.videoconfig']);
s = '';

nn = [];

vid = {};
while ~feof(fid)
  while isempty(strmatch('File',s)) & ~feof(fid)
    s = fgets(fid);
  end
  if ~isempty(strmatch('File',s))
    cc = strfind(s,'\');
    c = strfind(s,',');
    vid = cat(2,vid,{s(cc(end)+1:c(1)-1)});
    s = s(c(1)+1:c(2)-1);
    yed = str2num(s(1:4));
    mod = str2num(s(5:6));
    dad = str2num(s(7:8));
    hod = str2num(s(9:10));
    mid = str2num(s(11:12));
    sed = str2num(s(13:14));
    nn = cat(2,nn,datenum(yed,mod,dad,hod,mid,sed)+timeshift/24/60/60);
  end
end


fclose(fid);



for vi = 1:length(vid)
  file = vid{vi};
  
  dd = dir;
  [ye,mo,da,ho,mi,se] = datevec(nn(vi));
  rel_log = {};
  for i = 1:length(dd)
    if strmatch('classifier_log_',dd(i).name) 
      ss = dd(i).name(16:end-4);
      y = str2num(ss(1:4));
      m = str2num(ss(5:6));
      d = str2num(ss(7:8));
      if strcmp(dd(i).name(end-3:end),'.log') & ye==y & mo == m & da ==d & exist(['classifier_log_' ss '.out'],'file')
        rel_log = cat(2,rel_log,{ss});
      end
    end
  end
  nnn = [];
  for i = 1:length(rel_log)
    yea = str2num(rel_log{i}(1:4));
    mon = str2num(rel_log{i}(5:6));
    day = str2num(rel_log{i}(7:8));
    hou = str2num(rel_log{i}(10:11));
    minu = str2num(rel_log{i}(12:13));
    sec = str2num(rel_log{i}(14:15));
    
    nnn = cat(2,nnn,datenum(yea,mon,day,hou,minu,sec));
  end
  
    
  thresh = zeros(length(clas),2);

  for i = 1:length(clas)
    S = load(clas{i},'-mat');
    classifier = getfromdouble(S.classifier);
    thresh(i,:) = classifier.mapping;
  end
  
  
  
  
  inf = aviinfo(file);

  c = strfind(vid{vi},'.');
  if isempty(moviefile)
    file2 = [vid{vi}(1:c(end)-1), '_modified' vid{vi(c(end):end)}];
  else
    file2 = moviefile;
  end
  

  moviefil = avifile(file2,'fps',inf.FramesPerSecond,'Compression','DIV4');
  
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
  
   
   
  info = visualize_classifier(names,thresh,pospi,messages);
  
  finish = 0;
  framepos = 1;
  movframepos = 1;
  while ~finish 
    
    finish = 1;
    in = find(nnn<=nn(vi));
    if isempty(in)
      in = 1;
    else
      in = max(in);
    end
    log_file = sprintf('classifier_log_%s.out',rel_log{in});
    log_file2 = sprintf('classifier_log_%s.log',rel_log{in});
    
    
    
    S = load(log_file,'-mat');
    out = S.out;
    
    
    
    
    fid = fopen(log_file2,'r');
    s = fgets(fid);
    while isempty(strmatch('Start',s));
      s = fgets(fid);
    end
    fclose(fid);
    
    s = s(18:end);
    [ye,mo,da,ho,mi,se] = datevec(s);
    n = datenum(ye,mo,da,ho,mi,se)*24*60*60;
    
    nnnn = nn(vi)*24*60*60;
    for i = framepos:ceil(inf.NumFrames/idd)
      %for i = 1:20
      
      mov = aviread(file,(i-1)*idd+1:min(i*idd,inf.NumFrames-2));
      %   dat = repmat(uint8(zeros([round(sc*pos),3])),[1,1,1,length(mov)]);
      
      if sc~=1
        set(hh,'Visible','on');
        figure(hh);
        for j = movframepos:length(mov)
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
       nexti = 0;
       for j = movframepos:length(mov);
         movframepos = 1;
         printprogress((i-1)*idd+j,inf.NumFrames);    
         tip = ((i-1)*idd+j)/inf.FramesPerSecond+nnnn-n;
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
         elseif tip>size(out.x,2)
           finish = 0;
           movframepos = j;
           framepos = i;
           break;
         end
         moviefil= addframe(moviefil,dat);
           
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
       
     
     if finish == 0
       nnn = nnn(in+1:end);
       break;
     end
     
   end
   
end

  
moviefil = close(moviefil);
    
printprogress;

close all;