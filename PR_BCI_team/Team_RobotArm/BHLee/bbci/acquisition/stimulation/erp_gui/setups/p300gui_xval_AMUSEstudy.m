features= proc_jumpingMeans(epo, opt.selectival);
[features, opt.meanOpt] = proc_subtractMean(proc_flaten(features));
[features, opt.normOpt] = proc_normalize(features);
if do_xval,
    opt_xv= strukt('xTrials', [5 5], 'loss','classwiseNormalized');
    [dum,dum,outTe] = xvalidation(features, opt.model, opt_xv);
    me= val_confusionMatrix(features, outTe, 'mode','normalized');
    remainmessage = sprintf('Correct Hits: %2.1f, Correct Miss: %2.1f\n',100*me(1,1),100*me(2,2));
    xval_result = [100*me(1,1),100*me(2,2)];
end

    cls.C = trainClassifier(features,'FDshrink');
    epo = cntToEpo(Cnt,mrk,opt.ival);
    epo = proc_sortWrtStimulus(epo);
    epo = proc_baseline(epo,opt.baseline, 'beginning_exact');
    epo  = proc_selectChannels(epo, 'not', {'E*', 'Mas*'});
    fv= proc_jumpingMeans(epo, opt.selectival);
    fv = proc_subtractMean(proc_flaten(fv), opt.meanOpt);
    fv = proc_normalize(fv, opt.normOpt);
    out = apply_separatingHyperplane(cls.C, fv.x);
    out = reshape(out, 6, 15, []);
    corrDir = mrk.toe(find(mrk.y(1,:)))-10;
%     [clasDir dum] = find(mrk.y);
    corrDir = corrDir(1:15:length(corrDir));
%     corrDir = clasDir(find(clasDir > 8));
%     corrDir = corrDir(1:15:length(corrDir))-8;
%     clasDir(find(clasDir > 8)) = clasDir(find(clasDir > 8)) - 8;
%     clasDir = reshape(clasDir, 6,15,[]);
%     for i = 1:size(out, 2),
%       for j = 1:size(out, 3),
%         out(clasDir(:, i, j), i, j) = out(:, i, j);
%       end
%     end
  
    corrScore = {};
    incorrScore = {};
    winner = zeros(15, size(out, 3));
    for itId = 1:15,
        indSc = squeeze(median(out(:,1:itId, :), 2));
        [dum winner(itId, :)] = min(indSc, [], 1);
        corrId = winner(itId,:) == corrDir;
        incorrId = ~corrId;

        srtSc = sort(indSc, 1);
        srtSc = srtSc(2,:)-srtSc(1,:);

        corrScore{itId} = srtSc(corrId);
        incorrScore{itId} = srtSc(incorrId);
    end
    meanCorrSc = cellfun(@percentiles, corrScore, num2cell(repmat(50, 1,15)));
    percCorrSc = cellfun(@percentiles, corrScore, repmat({[2 98]},1,15), 'UniformOutput', false);
    percCorrSc = reshape([percCorrSc{:}], 2, 15);
  
    if any(cellfun(@isempty, incorrScore)),
        for itI = 1:length(incorrScore),
            if isempty(incorrScore{itI}),
                meanInCorrSc(itI) = 0;
                percInCorrSc(:,itI) = [0; 0];      
            else
                meanInCorrSc(itI) = percentiles(incorrScore{itI}, 50);
                percInCorrSc(:,itI) = percentiles(incorrScore{itI}, [2 98]);
            end
        end
    else
        meanInCorrSc = cellfun(@percentiles, incorrScore, num2cell(repmat(50, 1,15)));
        percInCorrSc = cellfun(@percentiles, incorrScore, repmat({[2 98]},1,15), 'UniformOutput', false);
        percInCorrSc = reshape([percInCorrSc{:}], 2, 15);
    end
  
    x = [1:15];
    p = polyfit(x, percInCorrSc(2,:), 3);
    newy = polyval(p, x);
    earlyDec = [];
    clear label totAbovThres
%     for i = 2:15,
%         stopId = find(corrScore{i} > newy(i));
%         newDec = ~ismember(stopId, earlyDec);
%         totAbovThres(i) = length(find(newDec));   
%         decTaken = winner(i,stopId(newDec));
%         if isempty(decTaken),
%             cDec = [];
%         else
%             cDec = decTaken == corrDir(stopId(newDec))';
%         end
%         label{i} = sprintf('%i\n%i / %i', totAbovThres(i), length(find(cDec)), length(find(~cDec)));
%         earlyDec = [earlyDec stopId(newDec)];
%     end  

% What do we need later?
newy = max(newy, meanCorrSc);
bbci.thresholds = newy;    