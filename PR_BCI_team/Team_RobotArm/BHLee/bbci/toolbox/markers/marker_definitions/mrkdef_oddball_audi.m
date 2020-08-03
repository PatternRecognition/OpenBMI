function mrk= mrkdef_oddball_audi(Mrk, file, varargin)
%MRKDEF_ODDBALL_AUDI - prepare markers for auditory oddball experiment
%
%Synopsis:
% MRK= mrkdef_oddball_audi(MRK, FILE, <OPT>)
%
%Arguments:
% MRK  - unprocessed markers e.g. es read in by readMarkerTable
% file - name of the eeg data file without file type ending, relative to         EEG_RAW_DIR
% OPT  - struct or property/value list of optional properties
% 
%Output:
% MRK - preprocessed narker structure with the following fields
%       .pos position of a marker (in number of samples)
%       .toe original coding of the mrks
%       .fs sample frequency
%       .y class assignment to classes named in:
%       .className (sequence is kept in .y columnwise) : {'tar' 'nontar'}
%       .ishit the stimulus was correctly identified as target or notarget
%       .duration time in msec from the stimulus presentation until the motor response
%       .add_res_pos positions of additional responses, when a response was already given within one run
%       .indexedByEpochs: contains mrk fields that have to be accounted for when making epochs.
%%
% Marker codings:
%   1 nontarget, i.e. standard
%   2 target, i.e. oddball
%  -8 response to nontarget
% -16 response to target
%
%
%See:
%   readMarkerTable, mrkdef_imag_fb01

%Author(s) Christine Carl, May-2006


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'markers_stddev', [1 2], ...
                  'markers_response', [-16 -8]);

fs = Mrk.fs;
Mrk2= readMarkerTable(file, 'raw');


Mrk = Mrk2;
Mrk.pos = ceil(Mrk.pos/Mrk2.fs*fs);
Mrk.fs = fs;

classDef= {opt.markers_response(1), opt.markers_response(2), ...
           opt.markers_stddev(1), opt.markers_stddev(2);
           'response_right','response_left','nontar','tar'};
mrk= makeClassMarkers(Mrk, classDef,0,0);
mrk2= makeClassMarkers(Mrk2, classDef,0,0);

mrord = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses','tar', 'nontar'),0,'remainclasses');
%ishit
mrres = mrk_sortChronologically(mrk_selectClasses(mrk, 'remainclasses','response_right','response_left'),0,'remainclasses');
mrres2 = mrk_sortChronologically(mrk_selectClasses(mrk2, 'remainclasses','response_right','response_left'),0,'remainclasses');
mrord2 = mrk_sortChronologically(mrk_selectClasses(mrk2, 'remainclasses','tar', 'nontar'),0,'remainclasses');
mrk.pos = [];mrk.toe = [];mrk.y = [];
mrk.ishit = []; mrk.duration = [];
mrk.add_res_pos =[];
for i = 1:length(mrord.pos)
    %find pos where real feedback starts so taht only those are considered later
  if i==length(mrord.pos), 
    pos1 = inf; 
  else
    pos1 = mrord.pos(i+1);
  end
  % only the first response is calculates as hit
  ind = find(mrres.pos>mrord.pos(i)&mrres.pos<pos1); 
  if isempty(ind), 
    continue; 
  elseif length(ind)>1
    mrk.add_res_pos =[mrk.add_res_pos mrres.pos(ind(2:end))];
  end
  
  pos2 = ind(1);
  
  % Distinguish initial trials and free gaming:
  
  if (mrres.pos(pos2)<pos1)
    mrk.pos = [mrk.pos, mrord.pos(i)];
    mrk.toe = [mrk.toe, mrord.toe(i)];
    mrk.y = [mrk.y, mrord.y(:,i)];
    mrk.ishit = [mrk.ishit,(mrord.y(:,i)'*mrres.y(:,pos2))>0];
    %duration should be in msec
    mrk.duration = [mrk.duration, mrres2.pos(pos2)-mrord2.pos(i)] ...
        / mrk2.fs*1000;
  end
end

% most probably the standards were confused with the nonstandards
if size(find(mrk.ishit==1))< size(find(mrk.ishit==0))
    mrk.ishit = ones(1,size(ishit,2))-mrk.ishit;
end


mrk.indexedByEpochs = {'ishit','duration'};
mrk = makeClassMarkers(mrk_sortChronologically(mrk),classDef);
mrk = mrk_selectClasses(mrk,{'tar', 'nontar'});