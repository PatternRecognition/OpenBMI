function mrk= getTuebingenEvents(file, targetDef, responseDef)
%mrk= getTuebingenEvents(file, targetDef, responseDef)

if ~exist('responseDef','var') | isempty(responseDef),
%  responseDef= {1, 1; 
%                [0.3 0.5], [0.1 0.3];
%                'left','right'};
  responseDef= {1, 1; 
                [1.5 2.5], [0.5 1.5];
                'left','right'};
end

% TTD Export marks only Target Codes, when the task differs from
% the preceeding task!
bot= readAlternativeMarkers(file, 'BCIState', {'BeginOfTrial';'bot'});
if isempty(bot.pos),
  bot= readMarkerTable(file, 100, 'BCIStateChange');
end
mrk= copyStruct(bot, 'toe', 'className');
mrk.toe= zeros(size(bot.toe));

mrk_alt= readAlternativeMarkers(file, 'BCIState', targetDef);
if isempty(mrk_alt.pos),
  mrk_alt= readAlternativeMarkers(file, 'BCIStateChange', targetDef);
end
cnt_marker= readGenericEEG(file, 'Marker', 500);
cnt_marker= proc_jumpingMax(cnt_marker, 5);
mrk_tap= getThresholdEvents(cnt_marker.x, responseDef, cnt_marker.fs);
trialLength= median(diff(bot.pos));

minDist= 4000/1000*bot.fs;

if ~isempty(mrk_tap.pos),
  mrk.tap.pos= NaN*ones(1,length(bot.pos));
  mrk.tap.pos2= NaN*ones(1,length(bot.pos));
  mrk.tap.pos3= NaN*ones(1,length(bot.pos));
end
for ei= 1:length(bot.pos),
  mi= max(find(mrk_alt.pos<=bot.pos(ei)));
  mrk.toe(ei)= mrk_alt.toe(mi);
  fi= find(mrk_tap.pos>bot.pos(ei)+minDist);
  if ~isempty(fi) & mrk_tap.pos(fi(1))<bot.pos(ei)+trialLength,
    mrk.tap.pos(ei)= mrk_tap.pos(fi(1));
    mrk.tap.toe(ei)= mrk_tap.toe(fi(1));   
    if length(fi)>1 & mrk_tap.pos(fi(2))<bot.pos(ei)+trialLength,
      if mrk_tap.toe(fi(2))~=mrk_tap.toe(fi(1)),
        fprintf('mistake in tapping sequence\n');
      else
        mrk.tap.pos2(ei)= mrk_tap.pos(fi(2));
      end
      if length(fi)>2 & mrk_tap.pos(fi(3))<bot.pos(ei)+trialLength,
        if mrk_tap.toe(fi(3))~=mrk_tap.toe(fi(1)),
          fprintf('mistake in tapping sequence\n');
        else
          mrk.tap.pos3(ei)= mrk_tap.pos(fi(3));
        end
      end
    end
  end
end
nClasses= size(targetDef,2);
mrk.y= zeros(nClasses, length(mrk.pos));
for ic= 1:nClasses,
  mrk.y(ic,:)= (mrk.toe==ic);
end
mrk.className= {targetDef{2,:}};

if isfield(mrk, 'tap'),
  nTapClasses= size(responseDef,2);
  mrk.tap.y= zeros(nTapClasses, length(mrk.tap.pos));
  for ic= 1:nTapClasses,
    mrk.tap.y(ic,:)= (mrk.tap.toe==ic);
  end
  mrk.tap.fs= mrk.fs;
  mrk.tap.className= {responseDef{3,:}};
end
