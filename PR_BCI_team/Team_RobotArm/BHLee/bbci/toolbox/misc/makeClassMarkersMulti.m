function mrk= makeClassMarkersMulti(mrk, classDef, simult, allow_repeats)
%mrk= makeClassMarkersMulti(mrk, classDef, <simult=80, allow_repeats=0>)
%
% IN   mrk             - event markers, see readMarkerTable
%      classDef        - cell array, each entry of the first row holds 
%                        the codes (toe) of the corresponding class
%                        a second row may contain class names
%      simult          - two events which less apart than simult [ms]
%                        are counted as simultanueos events,
%                        default is 80
%      allow_repeats   - repeats of one event is accepted as class
%
% OUT  mrk       struct for class markers
%         .pos       - marker positions [samples]
%         .toe       - type of event (marker code)
%         .fs        - sampling interval
%         .y         - class label
%         .className - class names, if given as second row in classDef
%
% SEE  readMarkerTable

if ~exist('simult', 'var'), simult=80; end
if ~exist('allow_repeats', 'var'), allow_repeats=0; end

select= find(ismember(mrk.toe, [classDef{1,:}]));
possel= mrk.pos(select);
toesel= mrk.toe(select);

nClasses= size(classDef,2);
class= zeros(1, length(select));
for ic= 1:nClasses,
  class(ismember(toesel, classDef{1,ic}))= ic;
end
if size(classDef,1)>1,
  mrk.className= {classDef{2,:}};
else
  mrk.className= cellstr(int2str((1:nClasses)'));
end

nSel= length(select);
mrk.pos= zeros(1, nSel);
mrk.toe= zeros(1, nSel);
mrk.y= zeros(nClasses, nSel);
group= {};
repeats.idx= [];
repeats.classes= [];
ie= 0;
im= 0;
while im<length(select),
  ie= ie+1;
  im= im+1;
  mrk.pos(ie)= possel(im);
  si= find(possel(im+1:end)-possel(im)<simult/1000*mrk.fs);
  if isempty(si),
    mrk.toe(ie)= toesel(im);
    mrk.y(class(im),ie)= 1;
  else
    multi= unique(class([im im+si]));
    ig= [];
    for ii= 1:length(group),
      if isequal(multi, group{ii}),
        ig= ii;
      end
    end
    if length(multi)==1,
      repeats.idx= [repeats.idx ie];
    end
    if isempty(ig),
      group= {group{:}, multi};
      mrk.y= [mrk.y; zeros(1,size(mrk.y,2))];
      ig= length(group);
      if length(multi)==1,
        mrk.className{nClasses+ig}= [classDef{2,multi} ' rep'];
        repeats.classes= [repeats.classes nClasses+ig];
      else
        str= sprintf('%s + ', classDef{2,multi});
        mrk.className{nClasses+ig}= str(1:end-3);
      end
    end
    pr= primes(10*max(multi));
    mrk.toe(ie)= -prod(pr(multi));
    mrk.y(nClasses+ig, ie)= 1;
    im= max(im+si);
  end
end

valid= 1:ie;
if ~allow_repeats,
  valid= setdiff(valid, repeats.idx);
  valid_classes= setdiff(1:size(mrk.y,1), repeats.classes);
  mrk.y= mrk.y(valid_classes,:);
  mrk.className= {mrk.className{valid_classes}};
end

mrk.pos= mrk.pos(valid);
mrk.toe= mrk.toe(valid);
mrk.y= mrk.y(:,valid);
