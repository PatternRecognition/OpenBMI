function show_continous_eegmovie(cnt,varargin);
%SHOW_CONTINUOUS_EEG visualizes EEG as a movie.
%
% usage:
%  (1)  show_continous_eeg(file,<opts>);
%  (2)  show_continous_eeg(cnt,<opts>);
%
% file -    name of file (global path or relative to EEG_MAT_DIR)
%       or  name of file.eeg (global path or relative to EEG_RAW_DIR)
% cnt  -    eeg file
% opts -    list of options or struct with fields
%        .mrk  -  mrk file to use in case (2)
%              or flag if markers should be shown in case (1)
%              or locations of marker .vmrk file in case (1)
%        .clab - name of channels to use (chanind) (default: all)
%        .ival - time interval to use in msec, use inf for the end of the file (default: [0 inf])
%        .steps - time steps between to frames in msec (default: 40)
%        .window_length - number of msec in one window (default: 5000)
%        .zeropoint - relative position of the current time point in window, leave empty if the zero-point should like the BrainVision Recorder (default: 0)
%        .range - range for each channel (default: [min all channels, max all channels], for separate channel ranges use a nx2 matrix!
%        .proc  - processing to use for the eeg-data in the form cnt = ...(cnt,...); ... (default: '')
%        .color - color for eeg (default: [0 0 1]), use nx3 matrix for different colors
%        .moviefile - name for the moviefile (empty: show only movie)
%
% Guido Dornhege, 23/04/2007

opt = propertylist2struct(varargin{:});

opt = set_defaults(opt,'mrk',1,...
                       'clab',[],...
                       'ival',[0 inf],...
                       'steps',40,...
                       'window_length',5000,...
                       'zeropoint',[],...
                       'range',[],...
                       'moviefile','',...
                       'color',[0 0 1],...
                       'proc','');
                       
if ischar(cnt)
  if length(cnt)>3 & strcmp(cnt(end-3:end),'.eeg')
    if ischar(opt.mrk)
      mrk= eegfile_readBVmarkers(opt.mrk);
    elseif opt.mrk
      mrk = eegfile_readBVmarkers(cnt(1:end-4));
    else
      mrk = struct;
    end
    cnt = eegfile_loadBV(cnt(1:end-4));
  else
    [cnt,mrk] = eegfile_loadMatlab(cnt);
    if ~opt.mrk
      mrk = [];
    end
  end
else
  if isstruct(opt.mrk)
    mrk = opt.mrk;
  else
    mrk = [];
  end
end

if ~isempty(opt.proc)
  eval(opt.proc);
end

if ~isempty(opt.clab);
  cnt = proc_selectChannels(cnt,opt.clab);
end

if isempty(opt.zeropoint)
  st = 0;
else
  st = opt.window_length*opt.zeropoint;
end

if opt.ival(1)>st
  cnt.x = cnt.x(min(1,round(opt.ival(1)-st/1000*cnt.fs)):end,:);
  if ~isempty(mrk)
    ind = find([mrk(:).pos]*1000/mrk(1).fs<opt.ival(1)-st);
    if length(mrk)>1
      mrk(ind) = [];
    else
      mrk.pos(ind) = [];
      try,mrk.toe(ind) = [];end
      try,mrk.y(:,ind) = [];end
    end
  end
end

if isempty(opt.zeropoint)
  en = opt.window_length*ceil(diff(opt.ival)/opt.window_length);
else
  en = diff(opt.ival)+opt.window_length*(1-opt.zeropoint);
end

if en<size(cnt.x,1)*1000/cnt.fs
  cnt.x = cnt.x(1:min(size(cnt.x,1),round(en/1000*cnt.fs)),:);
  if ~isempty(mrk)
    ind = find([mrk(:).pos]*1000/mrk(1).fs>en+opt.ival(1));
    if length(mrk)>1
      mrk(ind) = [];
    else
      mrk.pos(ind) = [];
      try,mrk.toe(ind) = [];end
      try,mrk.y(:,ind) = [];end
    end
  end
end

opt.ival(2) = min(opt.ival(2),opt.ival(1)+size(cnt.x,1)*1000/cnt.fs);

if isempty(opt.range)
  opt.range = [min(cnt.x(:)),max(cnt.x(:))];
end

if size(opt.range,1)==1
  opt.range = repmat(opt.range,[length(cnt.clab),1]);
end

if size(opt.color,1)==1
  opt.color = repmat(opt.color,[length(cnt.clab),1]);
end

clf;
if isempty(opt.moviefile)
  set(gcf,'DoubleBuffer','on');
else
  mov = avifile(opt.moviefile,'fps',1000/opt.steps);
end

p = zeros(1,length(cnt.clab));
zp = zeros(1,length(cnt.clab)+1);

for i = 1:length(cnt.clab)
  p(1,i) = subplot('position',[0.05,1-i/(length(cnt.clab)+1),0.95,0.99/(length(cnt.clab)+1)]);
  pp = plot((1:size(cnt.x,1))/cnt.fs*1000,cnt.x(:,i));
  set(pp,'Color',opt.color(i,:));
  set(p(i),'XTick',[],'YTick',[]);
  set(p(i),'XLim',[1,size(cnt.x,1)]/cnt.fs*1000,'YLim',opt.range(i,:));
  zp(i) = line([0 0],opt.range(i,:));
  set(zp(i),'Color',[1 0 0],'LineWidth',2);
  subplot('position',[0,1-i/(length(cnt.clab)+1),0.05,0.99/(length(cnt.clab)+1)]);
  axis off
  t = text(0,0,cnt.clab{i});
  set(t,'Rotation',90,'VerticalAlignment','top','HorizontalAlignment','center');
  t = text(0.2,-1,num2str(opt.range(i,1)));
  set(t,'VerticalAlignment','bottom','HorizontalAlignment','left');
  t = text(0.2,1,num2str(opt.range(i,2)));
  set(t,'VerticalAlignment','top','HorizontalAlignment','left');
  
  set(gca,'XLim',[0 1],'YLim',[-1 1]);
end

time = subplot('position',[0.05,0.2/(length(cnt.clab)+1),0.95,0.79/(length(cnt.clab)+1)]);
set(gca,'XTick',[ceil(opt.ival(1)/1000):opt.window_length/5000:floor(opt.ival(2)/1000)],'YTick',[]);
set(gca,'XLim', opt.ival/1000,'YLim',[0 1]);
if ~isempty(mrk)
  mrkpos = [mrk(:).pos];
  if length(mrk)>1
    mrkstr = {mrk(:).desc};
  else
    mrkstr = cellstr(num2str(mrk.toe'));
  end
  for i = 1:length(mrkpos)
    l = line(mrkpos(i)/mrk(1).fs*[1 1],[0 0.5]);
    t = text(mrkpos(i)/mrk(1).fs,0.55,mrkstr{i});
    set(t,'VerticalAlignment','bottom','HorizontalAlignment','center','FontUnits','normalized','FontSize',0.25);
  end
end
zp(end) = line([0 0],[0 1]);
set(zp(end),'Color',[1 0 0],'LineWidth',2);

if isempty(opt.zeropoint)
  tw = [0 opt.window_length];
else
  tw = [-opt.zeropoint,1-opt.zeropoint]*opt.window_length;
end


tw = tw+st;
tp = st;


waitForSync;
while tp <= diff(opt.ival)
  set(zp(1:end-1),'XData',tp*[1 1]);
  set(zp(end),'XData',(tp+opt.ival(1))/1000*[1 1]);
  set(p,'XLim',tw);
  set(time,'XLim',(tw+opt.ival(1))/1000);

  tp = tp+opt.steps;
  if isempty(opt.zeropoint)
    if tp>tw(end)
      tw = tw+opt.window_length;
    end
  else
    tw = [-opt.zeropoint,1-opt.zeropoint]*opt.window_length+tp;
  end
  drawnow;
  if isempty(opt.moviefile)
    waitForSync(opt.steps);
  else
    F = getframe(gcf);
    mov = addframe(mov,F);
  end
end

if ~isempty(opt.moviefile);
  mov = close(mov);
end

