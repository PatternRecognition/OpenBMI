function [dat]=func_pca(dat)
if isempty(dat)
    warning('[OpenBMI] Warning! data is empty.');
end

[T, nEvents , nChans]= size(dat.x);

%% reshaping the parameter as order*channel by trials
% Dat=permute(dat.x, [1,3,2]);
Dat=reshape(dat.x,[T* nEvents,nChans]);

b = mean(Dat,1);
B = repmat(b, [T*nEvents, 1]);
Dat = Dat - B;

C = cov(Dat);
[V, D] = eig(C);
[ev_sorted, sort_idx] = sort(diag(D), 'descend');
D = diag(ev_sorted);

%% whitening
V = V * diag(diag(D).^-0.5);

%% projecting the data with variance
dat.x=reshape(Dat,[T, nEvents,nChans]);
B = squeeze(repmat(b, [T,1,nEvents]));
B=reshape(B,[T, nEvents,nChans]);
dat.x=dat.x-B;
dat.x=func_projection(dat.x, V);



   


 
    
    
    
    
    
























