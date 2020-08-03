function [loss_ga lossAll t chanceAll] = proc_slidingClassificationAcc(epoList, opt)
% performs a classification on a sliding window of each epo, given in
% epoList and returns the grand-averaged loss at each averaged
% time-interval
% Johannes
%
% Added support for estimating chance level (Martijn)

if nargin<2
    opt = []
end
opt = set_defaults(opt, ...
    'ref_ival', [-150 0],...
    'cfy_ival', [0 800], ...
    'windowsize', 50, ... %in ms 
    'stepsize', 10, ...
    'loss', 'rocArea', ...
    'classificationMethod', 'FDshrink', ...
    'plotting', 0, ...
    'chance_est', 0);

vp=0
if ~iscell(epoList)
    epoList = {epoList};
end
for epo = epoList
    vp = vp+1;
    if ~isstruct(epo)
        epo = epo{1};
    end
    fv = proc_selectIval(epo, opt.cfy_ival);
    if ~isempty(opt.stepsize)
        dum = opt.cfy_ival(1):opt.stepsize:(opt.cfy_ival(2)-opt.windowsize);
        ivalMat = [dum' dum'+opt.windowsize];
        fv = proc_jumpingMeans(fv, ivalMat); %compute avg on a sliding window!
        t = mean(ivalMat,2);
    else
        fv = proc_jumpingMeans(fv, opt.windowsize/(1000/epo.fs)); %compute avg on a sliding window!
        t = fv.t;
    end
    
    opt_xv= strukt('xTrials', [3 3], 'loss', opt.loss, 'out_timing', 0, 'progress_bar', 0);
    
    ff= fv;
    
    for ii= 1:size(fv.x,1)
        ff.x = fv.x(ii,:,:) ;
        loss(ii)= xvalidation(ff, opt.classificationMethod, opt_xv);
        if opt.chance_est,
            warning(sprintf('Estimating chance level in %i shuffles. This may take time.', opt.chance_est));
            fr = ff;
            for jj = 1:opt.chance_est,
                fr.y = fr.y(:,randperm(size(fr.y,2)));
                chance(ii, jj) = xvalidation(fr, opt.classificationMethod, opt_xv);
            end
        end
    end
    
    if vp==1,
        lossAll= zeros(length(loss), length(epoList));
        chanceAll = zeros([size(chance) length(epoList)]) ;
    end
    lossAll(:,vp)= loss;
    chanceAll(:,:,vp) = chance;
end

loss_ga = mean(lossAll,2);


if opt.plotting
    %plot(fv.t, 100*lossAll, 'LineWidth',0.5);
    hold on;
    plot(fv.t, 100*mean(lossAll,2), 'LineWidth',2, 'Color','k');
    hold off;
    xlabel('time  [ms]');
    ylabel('classification loss  [%]');
end


