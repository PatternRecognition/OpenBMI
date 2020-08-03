function fv=proc_aar_online(cnt,varargin)
%
% fv =proc_aar_online(cnt,opt)
%
% IN: cnt   - continuous data struct
%        .x - data (samples x channels)
%        .clab - channel labels (referring to x)
%        .fs - sampling rate
%        
%     opt   - options with possible fields:
%        .state - internal state of AAR
%            .MOP - model order
%            .aarv - aar values
%            .H - start samples (previous iteration)
%            .V - variance of innovation process
%            .Z - a-priori state correlation matrix
%        .init - boolean: reset the state to opt.state.
%        
% OUT:fv    - struct containing field x: aar parameters.
%
% Author: Carmen Vidaurre, kraulem 12/07
% References: 

% TODO: if we run into performance problems: replace for-loop by matrix calculation.
% NOTE: maybe change fv from (samples x (mop*nchans)) to ((samples*mop) x nchans).
%how does cnt come?: each feature at at time is a row? yes.
% y has to be one channels at a time!!!
% cnt is all the channels? where do i get laplacian channels?
opt= propertylist2struct(varargin{:});
opt = set_defaults(opt,'init',false);
persistent state clab;
nchan=size(cnt.x,2);
if opt.init
  	if ~isfield(opt,'state')
    	error('need an initial state!');
  	end
  	if ischar(opt.state)
  		%state is stored in a file with name opt.state.
  		cini=load(opt.state);
  		%this file should contain:
  		%state{nchan} cells with fields MOP,UC,V,Z,aarv
  		%and also chorder, which should go to opt.
  		state=cini.state;
 		opt.chorder=state.clab;
  	else	
  		% state is the structure with initial conditions
  		state = opt.state;
		opt.chorder = state.clab;
  	end
	for ichan=1:nchan 
		itchan=chanind(cnt.clab,[opt.chorder{ichan} '*']); %itchan=real position of the first cini channel
		copystate.V{itchan}=state.V{ichan};
		copystate.aarv{itchan}=state.aarv{ichan};
		copystate.Z{itchan}=state.Z{ichan};
		state.H{ichan}=zeros(state.MOP,1);
		state.lastsample{ichan}=0;
		for ii = 1:state.MOP+1
			clab{(state.MOP+1)*(ichan-1)+ii} = sprintf('%s %i',cnt.clab{ichan}, ii);
		end	
	end;
end;
ovlimit=10; %!!atention, give as option?
MOP = state.MOP;
fv = copy_struct(cnt,'not','x','clab');
fv.x = zeros(size(cnt.x,1),nchan*(MOP+1));
fv.clab = clab;
%if signal exceeds some limits: DO NOT ADAPT, IT IS NOT EEG
if max(max(abs(cnt.x)))<ovlimit
  for ichan=1:nchan
    [aarv,state]=aar_single(cnt.x(:,ichan),state,ichan);
    fv.x(:,1+(ichan-1)*(MOP+1):ichan*(MOP+1)) = [aarv log(state.V{ichan})'];
  end;
else
  %PRINT A MSG OF NO ADAPTATION
  warning('aar_online:noAdaptation','threshold exceeded. No AAR adaptation.');
  for ichan=1:nchan
    fv.x(:,1+(ichan-1)*(MOP+1):ichan*(MOP+1)) = [ones(size(cnt.x,1),1)*state.aarv{ichan} log(state.V{ichan}(end))*ones(size(cnt.x,1),1)]; 
  end;
end;
%fv.x=fv.x';
return;
%------------------
function [z,state] = aar_single(y,state,ichan);

[nc,nr]=size(y);
UC=state.UC;
p = state.MOP;
e = zeros(nc,1);
V(1)=state.V{ichan}(end); %initialitation V
V0=V(1);
z = state.aarv{ichan}(ones(nc,1),:);
H=state.H{ichan};
lastsample=state.lastsample{ichan};
ESU = zeros(nc,1)+nan;

%------------------------------------------------
%	First Iteration
%------------------------------------------------
 
Z = state.Z{ichan};
zt= state.aarv{ichan};

%------------------------------------------------
%	Update Equations
%------------------------------------------------
        
for t=1:nc,
        %H=[y(t-1); H(1:p-1); E ; H(p+1:MOP-1)] 
        
        %if t<=p, H(1:t-1) = y(t-1:-1:1);     %H(p)=mu0;          % Autoregressive 
        %else     H(1:p) = y(t-1:-1:t-p); %mu0]; 
		%end;
		
		% I want this: H(1:p)=y(t-1:t-p);
        
        H(2:end)=H(1:end-1); %move all one sample        
        %at t=1, y(t-1) is from the previous call. 
        if t==1 
        	H(1)=lastsample; %take the last one from the previous call
        else
        	H(1) = y(t-1); %or just take the corresponding in this call
        end;
        
        % Prediction Error 
        E = y(t) - zt*H;
        
        e(t) = E;
        
        if ~isnan(E),
                E2 = E*E;
	        	AY = Z*H; 
                ESU(t) = H'*AY;
  				V0 = V0*(1-UC)+UC*E2;        
				V(t) = V0;      
                k = AY / (ESU(t) + V0);		% Kalman Gain
                zt = zt + k'*E;
       			W = 0.5*UC*(Z+Z');				%Vidaurre 2007
	        	Z = Z - k*AY';               % Schloegl 1998
    	else
				V(t) = V0;
    	end;     
		z(t,:) = zt;
		Z   = Z + W;               % Schloegl 1998
end;


state.aarv{ichan}=zt;
state.H{ichan}=H;
state.V{ichan}=V;
state.Z{ichan}=Z;
state.lastsample{ichan}=y(end);
return