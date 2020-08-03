function [dat,Wp,la]= proc_cspscp(dat, nComps, NORMALIZE)
%[epo, W, la]= proc_scpcsp(dat, <nComps=nChans/nClasses, NORMALIZE=0>);
%
% calculate variant of common spatial patterns (CSP) with objective to
% maximize the deflection from the baseline
% -> slow cortical potential (SCP) variantions
%
% IN   epo       - data structure of epoched data
%      nComps    - number of patterns to be calculated, deafult nChans/nClasses
%      NORMALIZE - normalize correlation matrix? default 0
%
% OUT  epo       - updated data structure
%      W         - CSP projection matrix
%      la        - eigenvalues of CSP projections (rows in W)

% by guido based on proc_csp

y= dat.x;
z= dat.y;
[T, nChans, nMotos]= size(y);

if ~exist('NORMALIZE', 'var') | isempty(NORMALIZE), NORMALIZE=0; end

if size(z,1)==1
  z= [z<0;z>0];
end

yy = squeeze(sum(y,1));
for t= 1:size(z,1),
  cl= find(z(t,:) >0);
  RR{t} = yy(:,cl)*yy(:,cl)'/length(cl);
end

R = zeros(nChans,nChans, size(z,1));
for t = 1:size(z,1),
   	if NORMALIZE,
	 	R(:,:,t)= RR{t}/trace(RR{t});
 	else
		R(:,:,t) = RR{t};
	end
end


[U,D]= eig(sum(R,3)); 

P= diag(1./sqrt(diag(D)))*U';
if size(z,1)==2
	[B,D]= eig(P*R(:,:,2)*P');
	W= B'*P;	
	if ~exist('nComps', 'var') | isempty(nComps),
  		fi= 1:nChans;
	else
  		[dd,di]= sort(diag(D));
   		fi= [di(1:nComps); di(nChans-nComps+1:end)];
  		la= [1-dd(1:nComps); dd(nChans-nComps+1:end)];
	end
	Wp= W(fi,:)';
else
	if ~exist('nComps','var') | isempty(nComps)
   		[B,D]= eig(P*R(:,:,1)*P');
   		Wp= B'*P;
	else
		Wp=[];
		for t = 1:size(z,1),
   			[B,D]= eig(P*R(:,:,t)*P');
   			W = B'*P;
   			[dd,di]= sort(-diag(D));
		        W = W(di(1:nComps),:);
   			Wp = [Wp;W];
		end  
	 	Wp = Wp';
	end
end


yo= zeros(size(y,1), size(Wp,2), nMotos);
for m= 1:nMotos,
  yo(:,:,m)= y(:,:,m)*Wp;
end

dat.x= yo;
dat.clab= cell(1, 2*nComps);
for cc= 1:2*nComps,
  dat.clab{cc}= sprintf('scp-csp %d', cc);
end
