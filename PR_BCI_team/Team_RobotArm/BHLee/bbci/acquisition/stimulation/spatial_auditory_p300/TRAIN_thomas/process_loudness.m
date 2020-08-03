function threshold=process_loudness(varargin)

global TODAY_DIR
% first get the file that should be analysed
if isempty(varargin),
[fname,fpath]=uigetfile('*.eeg','please specify the loudness-test eeg-file',TODAY_DIR);
loudfile=fullfile(fpath,[fname(1:end-4),'_values.mat']);
else
  loudfile=varargin{1};
  loudfile=[loudfile '_values.mat'];
end

if length(varargin)>1
visual=varargin{2};
else
  visual=0;
end
% check if the loudness-value file is also there
try
   
   load(loudfile);
catch
   error('the loudness-values were not found in the expected location');
end

% read the eeg-markers
eegfile=loudfile(1:end-11);
mrk= eegfile_readBVmarkers(eegfile);

%% process the data

% cut off everything up to the start marker (S 10)
% and after the end marker (S 17)

for i=1:length(mrk.desc),
  if strcmp(mrk.desc(i),'S 10'),
    s=i;
  end
  if strcmp(mrk.desc(i),'S 17'),
    e=i;
  end
  
end
mrk.desc=mrk.desc(1:e-1);
mrk.desc=mrk.desc(s+1:end);


speaker_loudness=[];
response=[];

for i=1:size(loudSeq,1),
value=sum(sum(loudSeq(i,:,:)));
speaker=find(sum(squeeze(loudSeq(i,:,:))));
speaker_loudness(i,speaker)=value;

end

while length(mrk.desc)>0,
  if strfind(mrk.desc{1},'S'),
    mrk.desc=mrk.desc(2:end);
    if length(mrk.desc)>0 & strfind(mrk.desc{1},'R'),
      mrk.desc=mrk.desc(2:end);
      response=[response ,1 ];
    else
      response=[response ,0 ];
    end
  end
  
  
end

if visual,
  figure;
  hold on
  for i=1:length(response),
    speaker=find(speaker_loudness(i,:));
    value=sum(speaker_loudness(i,:));
    if response(i),
      col='ob';
    else
      col='xr';
    end
    plot(value,speaker,col);
    
    
    
  end 
  grid on
end

% now find the quietest level that was heard for each speaker
% ignore any outliers

threshold=ones(1,6);

for i=1:length(response),
  if response(i),
  speaker=find(speaker_loudness(i,:));
  value=sum(speaker_loudness(i,:));
  if value<threshold(speaker),
    threshold(speaker)=value;
  end
  end
end
if visual,
  for i=1:length(threshold)
    plot(threshold(i),i,'og')
  end
end
  

