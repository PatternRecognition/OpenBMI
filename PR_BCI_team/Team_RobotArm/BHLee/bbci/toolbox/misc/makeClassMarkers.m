function mrk= makeClassMarkers(mrk, classDef, blockingTime, blockingDoubles)
%mrk= makeClassMarkers(mrk, classDef, blockingTime, <blockingDoubles=50>)
%
% IN   mrk             - event markers, see readMarkerTable
%      classDef        - cell array, each entry of the first row holds 
%                        the codes (toe) of the corresponding class
%                        a second row may contain class names
%      blockingTime    - events which follow up the preceeding event in
%                        less than blockingTime [ms] are rejected (label 0),
%                        default is 0
%      blockingDoubles - two events which less apart than blockingDoubles [ms]
%                        are counted as doubles and rejected (label 0),
%                        default is 50
%
% OUT  mrk       struct for class markers
%         .pos       - marker positions [samples]
%         .toe       - type of event (marker code)
%         .fs        - sampling interval
%         .y         - class label
%         .className - class names, if given as second row in classDef
%
% SEE  readMarkerTable

if ~exist('blockingTime', 'var'), blockingTime=0; end
if ~exist('blockingDoubles', 'var'), blockingDoubles=50; end

nClasses= size(classDef,2);

for ic = 1:nClasses
  if ischar(classDef{1,ic})
    if classDef{1,ic}(1)=='-'
      si = -1;
      str = classDef{1,ic}(2:end);
    elseif classDef{1,ic}(1)=='+'
      si = 1;
      str = classDef{1,ic}(2:end);
    else 
      si = 1;
      str = classDef{1,ic};
    end
    nu = 0;
    str = str([length(str):-1:1]);
    for i = 1:length(str);
      if str(i)=='*'
        nu = [nu,nu+2^(i-1)];
      elseif str(i) =='1'
        nu = nu+2^(i-1);
      elseif str(i)=='0'
      else 
        error('not understand');
      end
    end
    classDef{1,ic} = nu*si;          
  end  
end


select= find(ismember(mrk.toe, [classDef{1,:}]));
mrk.pos= mrk.pos(select);
mrk.toe= mrk.toe(select);

mrk.y= zeros(nClasses, length(select));
for ic= 1:nClasses,
  mrk.y(ic,:)= ismember(mrk.toe, classDef{1,ic});
end
if size(classDef,1)>1,
  mrk.className= {classDef{2,:}};
end

tooQuick= 1 + find(diff(mrk.pos)<blockingTime/1000*mrk.fs);
muchTooQuick= find(diff(mrk.pos)<blockingDoubles/1000*mrk.fs);
doubles= [muchTooQuick, muchTooQuick+1];

mrk.y(:,[tooQuick, doubles])= 0;
