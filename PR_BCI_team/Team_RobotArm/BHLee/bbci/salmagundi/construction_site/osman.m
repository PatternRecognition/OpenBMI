file = 'osman/subject2';

[cnt, mrk, mnt]= loadProcessedEEG(file);

epo= makeSegments(cnt, mrk, [400 1200]);
epo_lap = proc_selectChannels(epo,'C3','C4','CP3','CP4');
epo_lap = proc_baseline(epo_lap, [400 700]); 
fv= proc_selectIval(epo_lap, [750 1200]);   
fv= proc_jumpingMeans(fv, 5, 9);     

nTrials= [10 10]; 
msTrials= [3 10 round(9/10*size(epo.y,2))];    

% divide into train and test set
test = fv.x(:,:,find(sum(fv.y)==0));
fv.x(:,:,find(sum(fv.y)==0)) = [];
fv.y(:,find(sum(fv.y)==0)) = [];    % now fv is the train set

doXvalidation(fv, 'FisherDiscriminant', nTrials);

tiny_grid= sprintf('C3,_,C4\nCP3,legend,CP4');
mnt_tiny= setDisplayMontage(mnt, tiny_grid);

showERPgrid(fv, mnt_tiny);

% to classify the test set:
fv = proc_flaten(fv);
fis = train_FisherDiscriminant(fv.x,fv.y);
test = reshape(test,[size(test,1)*size(test,2), size(test,3)]);
out = sign(apply_separatingHyperplane(fis,test));
fprintf('There are %d left imagined movements and %d right imagined movements\n',sum(out==-1),sum(out==1));
		   
