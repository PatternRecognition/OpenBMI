function [blk,mnt] = getAugCogBlocks(filename,fs,flag);
%GETAUGCOGBLOCKS reads out the data of one experiment and prepares
%information
%
% usage: 
%      [blk,mnt] = getAugCogBlocks(file,<fs=100>);
% 
% input:
%      file : the name of the experiment, e.g. VPar_031205
%      fs   : sampling interval
%
% output:
%      blk   is a struct with fields
%       .fs    sampling interval
%       .ival  a 2xn matrix describing intervals regarding the
%              label
%       .y     a 10xn matrix with logical values describing the the
%              class regarding className
%       .className  a cell array with classNames
%       .name  the name of the experiment
%      mnt   a montage
%
% Guido Dornhege, 11/02/2004

if ~exist('filename','var') | isempty(filename)
  error('You have to specify the filename');
end

if ~exist('fs','var') | isempty(fs);
  fs = 100;
end


EEG_AUGCOG_DIR= '/home/neuro/data/AUGCOG/';

augcog_file= 'augcog_exp_database.txt';

[file, season, carf, audio, visual, calc,comb]= ...
    textread(augcog_file, '%s%f%s%s%s%s%s', 'delimiter',',');
file= deblank(file);
carf = deblank(carf);
audio = deblank(audio);
visual = deblank(visual);
calc = deblank(calc);
comb = deblank(comb);
season = num2cell(season);


augcog= struct('file', file, ...
               'season',season,...
               'carf', carf,...
	       'audio', audio,...
	       'visual',visual,...
	       'calc',calc,...
               'comb',comb);

ind = find(strcmp(filename,{augcog(:).file}));

if isempty(ind)
  error(['file does not exist, you should use one of', ...
         sprintf(' %s',augcog(:).file)]);
end

augcog = augcog(ind);


if augcog.season<=2
  mrk= readMarkerComments([EEG_AUGCOG_DIR 'season_' int2str(augcog.season) '/' augcog.file], fs);
elseif augcog.season==3
  task = {'Auditory','auditory',...
          'Visual','visual',...
          'Follow_car','carfollow',...
          'Calc','calc'};
  state = {'Base_on','base',...
           'Low_on','low',...
           'High_on','high'};
   
  markis = {task{1:2:end},state{1:2:end},'Off'};
  mrk = readMarkerTable([EEG_AUGCOG_DIR 'season_' int2str(augcog.season) '/' augcog.file],fs,markis,1:length(markis));
else 
  mrk = readMarkerTable([EEG_AUGCOG_DIR 'season_' int2str(augcog.season) '/' augcog.file],fs);
  classDef  = {1,2,3,4,6; 'high calc','high audio','base calc','base audio','off'};
  mrk= makeClassMarkers(mrk, classDef);

end
blk = struct('fs',mrk.fs);

switch augcog.season
 case 1
  
  % first session December 2003
  
  blk.className = {'low drive','high drive','low carfollow','high carfollow', ...
                   'low auditory','high auditory','low visual', ...
                   'high visual','low calc','high calc'};
  
  ranges = {};
  
  % goto to the second start
  
  c = find(strcmp(mrk.str,'start'));
  
  po = c(2)+1;
  state = 1;
  ap = c(2);
  an = {'carf','audio','visuell','rechnen'};
  
  
  while po<length(mrk.str)
    s = mrk.str{po};
    switch state
     case 1   % baseline
      switch s
       case 'rechnen'
        ranges = {ranges{:},[ap,po,1]};
        state = 2;ap = [];
       case 'ende'
        ranges = {ranges{:},[ap,po,1]};
        ap = [];
       case 'start'
        ap = po;
      end
     case 2
      switch s
       case 'rechnen'
        state = 3;
        ap = [];
        sit = 0;
      end
     case 3
      switch s
       case 'start'
        if isempty(ap)
          sit = 1;
          ap = po;
        end
       case 'carf'
        if sit == 5
          ranges = {ranges{:},[ap,po,3]};
        end
        ap = po;
        sit = 1;
       case 'audio'
        ap = po;
        sit = 2;
       case 'visuell'
        ap = po;
        sit = 3;
       case 'rechnen'
        ap = po;
        sit = 4;
       case 'ende'
        if ~isempty(ap) 
          ranges = {ranges{:},[ap,po,sit+2]};
          ap = [];
        end
       case 'switch'
        if sit == 5
          sit = 1;
        end
        
        while ~strcmp(an{sit},mrk.str{po}); 
          po = po+1;
        end
        
        if ~isempty(ap)
          ranges = {ranges{:},[ap,po,sit+2]};
          if sit == 1
            ap = po;
            sit = 5;
          else
            ap = [];
          end
          
        end
      end
     
    end
    po = po+1;
  end
  
  
  % fast is defined as time interval from the last start until ende
  % of the end of the file
  
  c = max(find(strcmp(mrk.str,'start')));
  d = max(find(strcmp(mrk.str,'ende')));
  if isempty(d) | d<c
    ranges = {ranges{:},[c,length(mrk.pos)+1,2]};
  else
    ranges = {ranges{:},[c,d,2]};
  end
  
  ranges = cat(1,ranges{:});
  mrk.pos(end+1) = inf;
  
  blk.ival = [mrk.pos(ranges(:,1));mrk.pos(ranges(:,2))];
  
  vec = zeros(4,1);
  
  blk.y = zeros(10,size(ranges,1));
  
  order = {augcog.carf,augcog.audio,augcog.visual,augcog.calc};
  for i = 1:size(ranges,1);
    switch ranges(i,3)
     case 1
      blk.y(1,i) = 1;
     case 2 
      blk.y(2,i) = 1;
     case {3,4,5,6}
      s = order{ranges(i,3)-2}(vec(ranges(i,3)-2)+1);
      vec(ranges(i,3)-2) = mod(vec(ranges(i,3)-2)+1,length(order{ranges(i,3)-2}));
      blk.y((ranges(i,3)-1)*2-(s=='e'),i)=1;
    end
  end
  
 case 2
  % second session march
  
  blk.className = {'low drive','low carfollow','high carfollow', ...
                   'low auditory','high auditory','low visual', ...
                   'high visual','low calc','high calc','low comb','high comb'};
  
  
  let = {augcog.carf,augcog.audio,augcog.visual,augcog.calc,augcog.comb};
  c = find(1-strcmp(mrk.str,'bem'));
  mrk.str= mrk.str(c);
  mrk.pos= mrk.pos(c);

  
  % combine all switches
  state = 0; % nothing started
  list = {};
  an = {'carf','audio','visual','calculate','comb'};
  for i = 1:length(mrk.str)
    str = mrk.str{i};
    switch state
     case 0
      if strcmp(str,'start')
        state = 1;
        list = cat(2,list,{'start';i});
        state2 = 0;
        modi = 0;
        switchon = 0;
        introd = 0;
        sit = 0;
      end
     case 1
      if strcmp(str,'end')
        if mod(sit,2)==1 & state2>0
          if isempty(ep), ep= i;end
          list = cat(2,list,{an{state2};ap},{'stop';ep});
        end
        state = 0;
        list = cat(2,list,{'end';i});
      end
      if state2==0
        c = find(strcmp(an,str));
        if ~isempty(c)
          state2 = c;
          ap = i;
          switchon = 0;
        end
      elseif state2<10
        c = find(strcmp(an,str));
        if ~isempty(c) 
          if c~=state2 & switchon==0;
            state2=5;
            ap = i;
          end
          if switchon==0
%            ap = i;
            modi = 1;
            introd = 1;
          end
          if switchon>=3
            ap = i;
            modi = 1;
            introd = 1;
            switchon = 0;
          end
        end
        if strcmp('switch',str) 
          if switchon==0
            ep = i;
          end
          switchon = switchon+1;
        end
        if switchon>0 & switchon<=2 & ~isempty(c) 
          list = cat(2,list,{an{state2};ap},{'stop';ep});
          sit = sit+1;
          introd = 0;
          ep = [];
          ap = i;
          switchon = 0;
        end
      end
    end
  end
  
  % transfer it
  
  d = find(strcmp('end',list(1,:)));
  d = d(1);
  c = find(strcmp('start',list(1,1:d)));
  c = c(end);
  blk.ival = [mrk.pos(c);mrk.pos(d)];
  blk.y = zeros(length(blk.className),1);
  blk.y(1) = 1;
  c = find(strcmp('start',list(1,:)));
  c = c(end);
  if c<size(list,2)-3
    c = size(list,2);
  end
  state = 0;
  j = d+1;
  while j<c
    switch state
     case 0
      if strcmp(list{1,j},'start')
        state = 1;
        poi = 0;
      end
     case 1
      if strcmp(list{1,j},'end')
        state = 0;
      else
        cc = find(strcmp(an,list{1,j}));
        if ~isempty(let{cc})
          blk.ival = cat(2,blk.ival,[mrk.pos(list{2,j});mrk.pos(list{2,j+1})]);
          blk.y = cat(2,blk.y,zeros(length(blk.className),1));
          ta = let{cc}(poi+1);
          poi = mod(poi+1,length(let{cc}));
          blk.y(2*cc+(ta=='d'),end) = 1;
        end
        j = j+1;
      end
    end
    j = j+1;
  end
   
  if c~=size(list,2)
    d = find(strcmp('end',list(1,c+1:end)));
    d = c+d(1);
    
    blk.ival = cat(2,blk.ival,[mrk.pos(list{2,c});mrk.pos(list{2,d})]);
    blk.y = cat(2,blk.y,zeros(length(blk.className),1));
    blk.y(1,end) = 1;
  end
  
 case 3
  % season 3 in June 2004
  task = {task{2:2:end}};
  state = {state{2:2:end}};
  blk.className = repmat(state,[1,length(task)]);
  blkcl = repmat(task,[length(state),1]);
  blkcl = blkcl(:);
  blk.className = strcat(blk.className,repmat({' '},[1,length(blk.className)]),blkcl');

  blk.y = [];
  blk.ival = [];
  ta = 0;
  st = 0;
  for i = 1:length(mrk.toe);
    to = mrk.toe(i);
    if to<=length(task)
      if st>0 & ~isempty(tiap)
        tiep = mrk.pos(i);
        blk.ival = cat(2,blk.ival,[tiap;tiep]);
        blk.y = cat(2,blk.y,zeros(length(blk.className),1));
        po = length(state)*(ta-1)+st;
        blk.y(po,end)=1;
        st = 0;
        tiap = [];tiep=[];
      end
      ta = to;
    elseif ta>0
      to = to-length(task);
      if to<=length(state)
        if st>0 & ~isempty(tiap) & st~=to
          tiep = mrk.pos(i);
          blk.ival = cat(2,blk.ival,[tiap;tiep]);
          blk.y = cat(2,blk.y,zeros(length(blk.className),1));
          po = length(state)*(ta-1)+st;
          blk.y(po,end)=1;
          st = 0;
          tiap = [];tiep=[];
        end          
        st = to;
        tiap = mrk.pos(i);
      elseif ~isempty(tiap)
        tiep = mrk.pos(i);
        blk.ival = cat(2,blk.ival,[tiap;tiep]);
        blk.y = cat(2,blk.y,zeros(length(blk.className),1));
        po = length(state)*(ta-1)+st;
        blk.y(po,end)=1;
        st = 0;
        tiap = [];tiep=[];
      end
      
    end
  end    
    
 case {4,5}

  blk.className = mrk.className(1:end-1);

  st = 0;
  blk.y = [];
  blk.ival = [];
  
  for i = 1:length(mrk.toe);
    ind = find(mrk.y(:,i));
    if ind<length(mrk.className)
      % start marker
      if st>0
        blk.y = cat(2,blk.y,zeros(length(mrk.className)-1,1));
        blk.y(st,end) = 1;
        blk.ival = cat(2,blk.ival,[start;mrk.pos(i)]);
      end
      st = ind;
      start = mrk.pos(i);
    else
      % off marker
      if st>0
        blk.y = cat(2,blk.y,zeros(length(mrk.className)-1,1));
        blk.y(st,end) = 1;
        blk.ival = cat(2,blk.ival,[start;mrk.pos(i)]);
      end
      st = 0;
      start = 0;
    end
  end
  
  
  
  
  
    
    
    
    
    
end
  
blk.name = [EEG_AUGCOG_DIR 'season_' int2str(augcog.season) '/' filename];

if nargout>1,
  clab = {'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', ...
          'F7', 'F8', 'T7', 'T8', 'MT1','MT2', 'Fz', 'Cz', 'Pz', ...
          'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', ...
          'TP9', 'TP10','Eog', 'Ekg1', 'Ekg2'};
  mnt= projectElectrodePositions(clab);
  mnt= setDisplayMontage(mnt, 'augcog');
end
  

