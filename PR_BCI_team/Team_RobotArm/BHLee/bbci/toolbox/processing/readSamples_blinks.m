function samp = readSamples_blinks(dat,iv, options)

% search in a given time series blinks. blinks are high peaks
% cnt datas
% iv interval where most artifacts find
% options: struct
% - range:range behind and before a given mark, where the
%      samples you get from (default [-300 300])
% - maxpeak: a given value a blink must go over (default: 1000)
% - nopeak: values where no peaks phases must be below (default: 200
% )
% - absolut: see for absolut value above and reflect. (default : 1
% for do it, 0 do it not)
% number: find only a given number of such blinks (default: [])
% channels: relevant channels (EOGv)


% check the input
if ~exist('options') | isempty(options)
  options = [-300 300];
end

if ~isstruct(options)
  options.range = options;
end

if ~isfield(options,'range')
  options.range = [-300 300];
end


if ~isfield(options,'maxpeak')
  options.maxpeak = 1000;
end

if ~isfield(options,'nopeak')
  options.nopeak = 200;
end

if ~isfield(options,'absolut')
  options.absolut = 1;
end

if ~isfield(options,'number')
  options.number = [];
end

if ~isfield(options,'channels') 
  options.channels = 'EOGv';
end

chan = find(strcmp(dat.clab,options.channels));



% frequencies
options.range = union(round(options.range*dat.fs/1000),[]);

ranges = [];
inter = iv.int{1};
for i = 1:size(inter,1)
  [ranges,IA,IB] = union(ranges,inter(i,1):inter(i,2));
end


if options.absolut
  data = abs(dat.x(ranges,chan));
else
  data = dat.x(ranges,chan);
end

nopeaks = data<options.nopeak;

[peak,index] = sort(-data);

point = [];
po = 1;

while peak(po)<-options.maxpeak & (isempty(options.number) | ...
				   length(point)<options.number)
  pea = index(po);
  fl = 1;
  for i = 1:length(point)
    if sum(nopeaks(union(point(i):pea,pea:point(i))))==0
       fl =0;
    end
  end
  if fl 
    point = cat(2,point,pea);
  end
  po = po+1;
end

point = ranges(point);


samp = [];
for i = point
  sam = dat.x(i+(options.range(1):options.range(2)),:);
  if dat.x(i,chan)<0 fac = -1; else fac = 1; end
  samp = cat(3,samp,fac*sam);
end

  










