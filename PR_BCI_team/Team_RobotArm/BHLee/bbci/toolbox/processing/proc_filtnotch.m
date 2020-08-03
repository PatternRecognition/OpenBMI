%function dat=proc_filtnotch(cnt,<fa,fg,r,pl>);
%
%IN   cnt - EEG data structure
%     fa  - Sampling Frequency; Default: 100Hz
%     fg  - Frequency to erase; Default: 50Hz
%     r   - Default: 0.98
%           0<r<1: Fastest Version. A high value represents a filter
%           with a small number of frequencies being effected and a small
%           damping
%         - -50<r<-17 It ist possible to choose the damping at the
%           choosen frequency within these values 
%         - 1<r<1522 If you choose a value bigger than 1, the Cut-Off-Frequencies
%           will have the distance r (in Hz)
%
%     pl  - If pl=1, the amplitude- and the phasediagram of the used filter will be
%           plotted in a seperated figure
%           Default: pl=0
%OUT  dat - updated data structure
%
%proc_FiltNotch filters the data structure cnt on a specific frequency, especially
%to erase harmonic 50Hz parts from the Signals. It works with a \
%z-transformed filterfunction
%with 6 coefficients.
%
% Kai Melhorn, 14/07/2004

function [dat]=proc_filtnotch(dat,fa,fg,r,pl);

%warning off MATLAB:fzero:UndeterminedSyntax

 
if ~exist('fg','var') | isempty(fg), fg=50;  end
if ~exist('fa','var') | isempty(fa), fa=100; end
if ~exist('r','var')  | isempty(r),  r=0.98; end
if ~exist('pl','var') | isempty(pl), pl=0;   end

[B, A]= proctuil_filtnotch(fa, fg, r);

if pl==1;
  figure;
  [H,F]=freqz(B,A,[],fa);
  freqz(B,A);
end


mittel = mean(dat.x,1);
dat.x = dat.x-repmat(mittel,[size(dat.x,1),ones(1,ndims(dat.x)-1)]);

%proc_filt
dat=proc_filt(dat,B,A);

%Mean-values are added to the filtered signals
dat.x = dat.x+repmat(mittel,[size(dat.x,1),ones(1,ndims(dat.x)-1)]);
