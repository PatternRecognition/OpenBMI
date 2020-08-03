function mrk = mrkdef_imag_fb04(Mrk,file,opt);
%MRKDEF_IMAG_FB02 - prepare markers for extract marker information from auditory 1D fixed duration feedback (feedback_cursor_1d_pro)
%
%Synopsis:
% MRK= mrkdef_imag_fb04(MRK,FILE,OPT)
%
%Arguments:
% MRK  - unprocessed markers e.g. es read in by readMarkerTable
% file - name of the eeg data file without file type ending, relative to         EEG_RAW_DIR
% OPT  - struct of optional properties
% 
%Output:
% MRK - preprocessed narker structure with the following fields
%       .pos position of a marker (in number of samples)
%       .toe original coding of the mrks
%       .fs sample frequency
%       .y class assignment to classes named in:
%       .className (sequence is kept in .y columnwise) : {'left' 'right' 'rotation'}
%       .ishit the stimulus was correctly identified as target or notarget, rejected files are considered as misses
%       .free pos where cursor can be moved in a trial
%       .isrejected trials where cursor was in the grey rectangle at the end of the trial, i.e. cursor was to close to initial starting point to count this trial as a miss or hit
%       .duration time in msec from the stimulus presentation until end of trial
%       .indexedByEpochs: contains mrk fields that have to be accounted for when making epochs.
%       .adaptation_trial if these trials should be omitted in the analysis, they have to be taken out explicitely
%       .run_no: number of feedback and classifier log files 
%       .cursor_on 1 if cursor on
%%
% Marker codings:
%   see feedback_cursor_1d_pro
%
%   opt.cursor_on indicates if cursor is visible or not
% ATTENTION: here the markers are not compared and justified from feedback and classififier logfiles
%
%
%See:
%   readMarkerTable, mrkdef_imag_fb01

%Author(s) Christine Carl, May-2006


%% ----- extract marker information from 1D fixed duration
%% ----- initial adaptation cursor off feedback.

%% cursor on or off is indicated by opt.cursor_on.

opt.cursor_on = 0;
opt.adaptation = 1;
opt.markertypes = [1 2 11 12 21 22 25 60 70 71 200 210 220 230]
global problem_marker_bit

Mrk2= readMarkerTable(file, 'raw');
fs = Mrk.fs;
if ~isfield(opt,'logf')
  if strcmp(opt.marker_type,'response')
    % This is a hack and will most probably not work for 2 players.
    Mrk2.toe = abs(Mrk2.toe);
  end
  %[Mrk2,logf,flogf] = extract_logfiles(Mrk2,file,opt);
else
  logf = opt.logf;
  flogf = opt.flogf;
end

% the run number is determined by the filename.
if file(end)>'1'&file(end)<='9'
  run_baseidx = 1+str2num(file(end));
else
  run_baseidx = 0;
end

Mrk = Mrk2;
Mrk.pos = ceil(Mrk.pos/Mrk2.fs*fs);
Mrk.fs = fs;

classDef = {1,2,11,12,21,22,25, 60,70,71,200, 210,211,212,213 ; 
            opt.classes_fb{:}, ...
            ['hit ' opt.classes_fb{1}], ['hit ' opt.classes_fb{2}], ...
            ['miss ' opt.classes_fb{1}], ['miss ' opt.classes_fb{2}], ...
            'reject',...
            'free', 'rotation', 'rotation off','init',...
            'game_play','game_pause', 'game_stop', 'game_end'};
            
%different temporal resolution
mrk= makeClassMarkers(Mrk, classDef,0,0);
%mrk2 only needed to calculate exact mrk.duration
mrk2= makeClassMarkers(Mrk2, classDef,0,0);

%  mrk.log = logf;
%left vs right
mrord = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses',opt.classes_fb),0,'remainclasses');
%ishit
mrres = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses','hit*','miss*','reject'),0,'remainclasses');
%freeing og mrk
mrfree = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses','free*'),0,'remainclasses');
% same for mrk2
mrord2 = mrk_sortChronologically(mrk_selectClasses(mrk2,'remainclasses', opt.classes_fb),0,'remainclasses');
mrres2 = mrk_sortChronologically(mrk_selectClasses(mrk2, 'remainclasses','hit*','miss*','reject'),0,'remainclasses');

mrrot = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses','rotation*',0,'remainclasses'));


mrk.className = {mrord.className{:},'rotation'};
mrk.pos = [];mrk.toe = [];mrk.y = [];
mrk.ishit = []; mrk.free = []; mrk.isrejected =[]; mrk.duration = [];
mrk.adaptation_trial = [];

% for initialization only 
for ii = 1:2:length(mrrot.pos)
  % initial rotation phase.
  mrk.pos = [mrk.pos, mrrot.pos(ii)];
  mrk.toe = [mrk.toe, mrrot.toe(ii)];
  mrk.y = [mrk.y, [0;0;1]];
  mrk.ishit = [mrk.ishit, 0];
  mrk.isrejected = [mrk.isrejected, 0];
  mrk.free = [mrk.free, mrrot.pos(ii)];
  mrk.duration = [mrk.duration mrrot.pos(ii+1)-mrrot.pos(ii)];
  mrk.adaptation_trial = [mrk.adaptation_trial 1];
end

for i = 1:length(mrord.pos)
    %find pos where real feedback starts so taht only those are considered later
  if i==length(mrord.pos), 
    pos1 = inf; 
  else
    pos1 = mrord.pos(i+1);
  end
  ind = find(mrres.pos>mrord.pos(i)); 
  if isempty(ind), continue; end
  pos2 = ind(1);
  ind = find(mrfree.pos>mrord.pos(i)); 
  if isempty(ind), continue; end
  pos3 = ind(1);
  % Distinguish initial trials and free gaming:
  init_ind = find(mrfree.y(:,pos3));
  
  if mrfree.pos(pos3)<mrres.pos(pos2) & mrres.pos(pos2)<pos1 & ...
        (pos3==length(mrfree.pos) | mrfree.pos(pos3+1)>mrres.pos(pos2))
    mrk.pos = [mrk.pos, mrord.pos(i)];
    mrk.toe = [mrk.toe, mrord.toe(i)];
    mrk.y = [mrk.y,[mrord.y(:,i);0]];
    mrk.ishit = [mrk.ishit, ([1, 1, 0 0 0]*mrres.y(:,pos2))>0];
    mrk.isrejected = [mrk.isrejected, ([0 0 0 0 1]*mrres.y(:,pos2))>0];    mrk.free = [mrk.free, mrfree.pos(pos3)];
    %duration should be in msec
    mrk.duration = [mrk.duration, mrres2.pos(pos2)-mrord2.pos(i)] ...
        / mrk2.fs*1000;
    if strcmp(mrfree.className{init_ind},'free')
      mrk.adaptation_trial = [mrk.adaptation_trial 0];
    else
      mrk.adaptation_trial = [mrk.adaptation_trial 1];
    end
  end
end
mrk.indexedByEpochs = {'ishit','isrejected','free','duration','run_no','adaptation_trial','cursor_on'};

mrk.run_no = zeros(1,length(mrk.pos));
mrk.run_no = mrk.run_no+run_baseidx;
mrk.cursor_on = ones(size(mrk.pos))*opt.cursor_on;


mrk = makeClassMarkers(mrk_sortChronologically(mrk),classDef);
mrk = mrk_selectClasses(mrk,{opt.classes_fb{:},'rotation'});

mrk.flogf = [];
mrk.logf = [];

