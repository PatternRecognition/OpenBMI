nn= 1;  %% files: 1, 2 or 3

export_list= {'basket_3_targets', 
              'basket_6_targets',
              '1d_relative'};

global EEG_EXPORT_DIR

file= [EEG_EXPORT_DIR 'for_florin_' export_list{nn}];
load(file);

co= chanind(cnt, 'out');
cy= chanind(cnt, 'yData');

integrate= 5;
bias= -2;
scale= 0.1;
dist= 0.25;
range= [-1 1];
damping= 100;
trial_idx= 53:252;

if nn==2,
  dist= 0;
  damping= 130;
end

if nn==3,  %% this is for the 1d relative experiment
  cy= chanind(cnt, 'xData');
  mrk= mrk_selectClasses(mrk, 'left','right','foot');
  bias= 0;
  scale= 0.1;
  dist= 0.2;
  range= [-0.75 0.75];
  damping= 60;
  trial_idx= 77:200;  %% trials have different length!
end

yyy= cnt.x(:,cy);

soo= cnt.x(:,co); 
soo(find(isnan(soo)))= 0;

%% smoothing 
soo= (movingAverage(soo, integrate*4, 'causal') + bias ) * scale;

%% dead zone
iPos= find(soo>0);
iNeg= find(soo<0);
soo(iPos)= max(0, (soo(iPos)-dist)/(1-dist));
soo(iNeg)= -max(0, (-soo(iNeg)-dist)/(1-dist));

%% range
soo(find(soo>range(2)))= range(2);
soo(find(soo<range(1)))= range(1);

first_good= min(find(~isnan(yyy)));
k= max(find(mrk.pos<first_good));

%% repeat this line
k=k+1; plot([cumsum(soo(mrk.pos(k)+trial_idx))/damping, yyy(mrk.pos(k)+trial_idx)])
