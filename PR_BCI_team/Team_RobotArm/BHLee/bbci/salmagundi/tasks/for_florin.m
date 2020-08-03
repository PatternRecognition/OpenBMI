load('for_florin_basket_3_targets');
%% -> cnt: data structure
%%    mrk: marker structure
%%    mnt: electrode montage (.x .y: 2d projected positions)

%% see electrodes:
text(mnt.x, mnt.y, mnt.clab)
set(gca, 'xLim',[-1.2 1.2], 'yLim',[-1.2 1.2])


%% basic script for plotting average trajectories

%% make epochs of 2.5 seconds
iv= 0:2500*cnt.fs/1000;
[nClasses, nEvents]= size(mrk.y);
T= length(iv);
IV= iv(:)*ones(1,nEvents) + ones(T,1)*mrk.pos;

%% classifier output is the last but 2 channel
out= reshape(cnt.x(IV,end-2), size(IV));

out_avg= zeros(T, nClasses);
for cc= 1:nClasses,
  idx= find(mrk.y(cc,:));
  out_avg(:,cc)= mean(out(:,idx),2);
end

plot(out_avg);
