dateStr= '01_07_24';
paceStr= '1s';

%% timepoint of classification
toc= -120;

classy= 'FisherDiscriminant';
nTrials= [10 10];

file= ['Gabriel_' dateStr '/selfpaced' paceStr 'Gabriel'];
[cnt, mrk, mnt]= loadProcessedEEG(file);
epo= makeSegments(cnt, mrk, [-1300 0] + toc);

%% for unbalanced transition matrices
equi_idcs= getEventPairs(mrk, paceStr);
[divTr, divTe]= sampleDivisions(epo.y, nTrials);
for ii= 1:length(divTr),
  for jj= 1:length(divTr{ii}),
    grouped= metaintersect(equi_idcs, divTr{ii}{jj});
    grouped= equiSubset(grouped);
    divTr{ii}{jj}= [grouped{:}];
  end
end
epo.divTr= divTr;
epo.divTe= divTe;

%fv= proc_selectChannels(epo, 'FC#', 'C#', 'CP#');
fv= proc_selectChannels(epo, 'FC5-6', 'C5-6', 'CCP5-6', 'CP5-6');
fv= proc_filtBruteFFT(fv, [0.8 3], 128, 150);
fv= proc_jumpingMeans(fv, 5);

[em,es,out,avErr,evErr]= doXvalidationPlus(fv, classy, nTrials);
fprintf('%.1f%% were always classified correctly\n', ...
        100*sum(evErr==0)/length(evErr));
fprintf('%.1f%% were never classified correctly\n', ...
        100*sum(evErr==1)/length(evErr));
outliers= find(evErr>0.1);
fprintf('%.1f%% were misclassified in > 10%% of the validation trials:\n', ...
        100*length(outliers)/length(evErr));


global EEG_RAW_DIR
if file(1)==filesep,
  fullName= file;
else
  fullName= [EEG_RAW_DIR file];
end

fid= fopen([fullName '.vmrk'], 'r');
%% read header up to the marker infos
keyword= '[Marker Infos]';
header= '';
headerLines= 0;
ok= 0;
while ~ok & ~feof(fid),
  str= fgets(fid);
  header= [header str];
  headerLines= headerLines + 1;
  ok= strncmp(keyword, str, length(keyword));
end
ok= 0;
str= '';
while ~ok & ~feof(fid),
  header= [header str];
  str= fgets(fid);
  headerLines= headerLines + 1;
  ok= str(1)~=';';
end
headerLines= headerLines - 1;
if ~ok,
  error('section [Marker Info] not found in header file');
end
fclose(fid);


[dummy,dummy,orig_fs]= readGenericHeader(file);
[mrkno,mrktype,desc,pos,pnts,chan,seg_time]= ...
    textread([fullName '.vmrk'], 'Mk%u=%[^,]%s%u%u%u%s', ...
             'headerlines', headerLines, 'delimiter',',');

nEvents= length(mrk.pos);
%mrkno= [mrkno; zeros(nEvents,1)];
for ie= 1:nEvents,
  mrktype{end+1}= 'Response';
  desc{end+1}= sprintf('R%3d', round(100-100*evErr(ie)));
end
pos= [pos; round(mrk.pos*orig_fs/mrk.fs)' + toc];
pnts= [pnts; ones(nEvents,1)];
chan= [chan; zeros(nEvents,1)];
blanx= cell(nEvents, 1);
[blanx{:}]= deal('');
seg_time= cat(1, seg_time, blanx);

[so,si]= sort(pos);


fid= fopen([fullName '_cs.vmrk'], 'wt');
if fid==-1,
  error('cannot write new marker file');
end
fprintf(fid, header);
for im= 1:length(pos),
  in= si(im);
  fprintf(fid, 'Mk%u=%s,%s,%u,%u,%u,%s\n', im, mrktype{in}, desc{in}, ...
          pos(in), pnts(in), chan(in), seg_time{in});
end
fclose(fid);
