function bbci = bbci_train_rankdiff(bbci, data),

opt = propertylist2struct(bbci.calibrate.early_stopping_param{:});
mrk = data.mrk;
opt.nIters = opt.nIters / size(opt.indices, 2);

% need to rerun the xval with no rejections
fv= cntToEpo(data.cnt, mrk, bbci.feature.ival, ...
             'clab',bbci.signal.clab);
fv= bbci_calibrate_evalFeature(fv, bbci.feature);

opt_xv= strukt('sample_fcn',{'chronKfold', 10}, ...
               'loss','classwiseNormalized', ...
               'verbosity', 1);
[loss,dum,cls]= xvalidation(fv, bbci.calibrate.settings.model, opt_xv);

% since we can not guarantee that the number of classes doesn't change, we 
% have to sort by hand here. warning, ugly for loop!
out = ones(opt.nClasses, opt.nIters, max(mrk.trial_idx))*NaN;
trg = zeros(size(opt.indices,2), max(mrk.trial_idx));
for trial = 1:max(mrk.trial_idx),
    idx = find(mrk.trial_idx==trial);
    nClasses = length(idx)/opt.nIters;
    block_cnt = 1;
    trgs = mrk.stimulus(-1+idx(1)+find(mrk.y(1,idx(1:nClasses))));
    for level = 1:size(opt.indices,2),
        trg(level,trial) = trgs(ismember(trgs, opt.indices{1,level}));
    end
    for sub_idx = 1:length(idx),
        subtrial = idx(sub_idx);
        out(mrk.stimulus(subtrial), block_cnt, trial) = cls(subtrial); 
        if ~mod(subtrial, nClasses), 
            block_cnt = block_cnt +1;
        end
    end
end

winner = zeros(max(mrk.block_idx), size(opt.indices, 2), max(mrk.trial_idx));
second = winner;
dist = winner;
rnk_diff = zeros(max(mrk.block_idx),max(mrk.trial_idx));

corrScore =ones(max(mrk.block_idx),max(mrk.trial_idx))*NaN;
incorrScore = corrScore;
% prepare the distributions 
for itId = 1:max(mrk.block_idx),
    indSc = squeeze(median(out(:,1:itId,:),2));
    for level = 1:size(opt.indices, 2),
        [srt srtId] = sort(indSc(opt.indices{1,level},:));
        winner(itId, level, :) = srtId(1,:) + (-1+min(opt.indices{1,level}));
        second(itId, level, :) = srtId(2,:) + (-1+min(opt.indices{1,level}));
        dist(itId, level, :) = srt(2,:) - srt(1,:);
    end
    [dum rorcol] = min(dist(itId, :, :),[],2);
    rorcol = squeeze(rorcol);
    second_best = winner;
    for tr = 1:max(mrk.trial_idx),
        id = rorcol(tr);
        second_best(itId,id,tr) = second(itId,id,tr);
        rnk_diff(itId,tr) = diff([mean(indSc(winner(itId,:,tr),tr)), mean(indSc(second_best(itId,:,tr),tr))]);
    end

    corrId = min(squeeze(winner(itId,:,:)) == trg);
    
    corrScore(itId,corrId) = rnk_diff(itId,corrId);
    incorrScore(itId,~corrId) = rnk_diff(itId,~corrId); 
end

% assume we take R to be 1
R = .5;
bbci.feedback.early_stopping.active = 1; 
thresholds = max(max(max(incorrScore,[],2)',0), ...
            R*nanmedian(corrScore,2)');
bbci.feedback.early_stopping.param = reshape(repmat(thresholds,2,1),1,[]);

end