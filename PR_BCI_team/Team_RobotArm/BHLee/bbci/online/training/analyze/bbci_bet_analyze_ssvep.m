% input:
%   bbci   struct, the general setup variable from bbci_bet_prepare
%   opt    a struct with fields
%       nPat    the number of patterns to use
%       usedPat the used Pattern (only for choose of the features)
%       clab    the used channels
%       ival    the train interval
%       band    the used band
%       filtOrder   the filt Order
%       ilen_apply the apply length
%       dar_ival    the ival the plots are visulized
%       model   the model
%   Cnt, mrk, mnt
%
% output:
%   analyze  struct, will be passed on to bbci_bet_finish_ssvep
%

PRESELECT_CLASSES=0;
USEALLHARMONICS=0;
NHarmonics = 2;
opt.CSP4EACHFREQ=1;
Nclasses=8;
XVAL=0;
withgraphic=1;
inside=1;
train_ival=4000;
%for iii=1:10
%opt.filter_width=0.2*iii
opt.filter_width=0.5;


% x-val options:
proc= struct('memo', 'csp_w');
proc.train= ['global hlp_w; ' ...
	     '[fv,csp_w]= proc_csp3(fv); ' ...
	     'fv= proc_variance(fv); ' ...
	     'fv= proc_logarithm(fv);'];
proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ' ...
	     'fv= proc_variance(fv); ' ...
	     'fv= proc_logarithm(fv);'];

analyze = [];
clear features
try
  cnt0=Cnt;
  clear Cnt
catch
end
%some plotting opts:
grd= sprintf('scale,FC1,FCz,FC2,legend\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4');
mnt= mnt_setGrid(mnt, grd);
mnt_spec= mnt;
mnt_spec.box_sz= 0.9*mnt_spec.box_sz;

colOrder= [[1 0 0];[0 0.7 0];[0 0 1];[0 1 1];[1 0 1]; [1 1 0]];
grid_opt= struct('colorOrder', colOrder);
grid_opt.scaleGroup= {scalpChannels, {'EMG*'}, {'EOG*'}};

spec_opt= grid_opt;
spec_opt.yUnit= 'power';
spec_opt.xTickMode= 'auto';
spec_opt.xTick= 10:5:30;
spec_opt.xTickLabelMode= 'auto';
rsqu_opt= {'colorOrder','rainbow'};

fig_opt= {'numberTitle','off', 'menuBar','none'};

if ~isfield(opt, 'usedPat')
  opt.usedPat = 1:2*opt.nPat;
end


labels={'4 Hz','5 Hz','6 Hz','7.5 Hz','8 Hz','10 Hz','12 Hz','15 Hz'};
freq_array=[4 5 6 7.5 8 10 12 15];
stimDef= {1, 2, 3, 4, 5, 6, 7, 8, [100 249 253];
	  labels{:}, 'stop'};

mrk= mrk_defineClasses(mrk_orig, stimDef);

mnt= getElectrodePositions(cnt0.clab);
grd= sprintf(['C5,C3,C1,Cz,C2,C4,C6\n' ...
	      'CP5,CP3,CP1,CPz,CP2,CP4,CP6\n' ...
	      'P5,P3,P1,Pz,P2,P4,P6\n', ...
	      'PO7,PO3,OPO1,POz,OPO2,PO4,PO8\n', ...
	      'scale,O1,OI1,Oz,OI2,O2,legend']);
mnt= mnt_setGrid(mnt, grd);

% use a method to take out noisy trials (artifact rejection)

blk= [];
for ff= 1:Nclasses,
  %cn= sprintf('freq %02d', ff)
  cn=char(mrk.className{ff});
  blk0= blk_segmentsFromMarkers(mrk, ...
				'start_marker',cn, ...
				'end_marker','stop');
  blk0.className= {cn};
  blk= blk_merge(blk, blk0);
end
[cnt, blkcnt]= proc_concatBlocks(cnt0, blk);

mkk= mrk_evenlyInBlocks(blkcnt, train_ival);
if withgraphic
[mkk, rClab]= reject_varEventsAndChannels(cnt, mkk, [0 train_ival-1],'visualize', 1);
else
[mkk, rClab]= reject_varEventsAndChannels(cnt, mkk, [0 train_ival-1],'visualize', 0);
end

if withgraphic,
  cnt_flt= proc_localAverageReference(cnt, mnt);
  epo=makeEpochs(cnt_flt,mkk,[0 train_ival-5]);
  spec=proc_spectrum(epo,[5 40]);
  figure;
  grid_plot(spec,mnt,spec_opt); 
end

%cnt=proc_selectChannels(cnt,'not',rClab);
cnt=proc_selectChannels(cnt,'not','F*','T*','C*','Ref');
mnt_csp= getElectrodePositions(cnt.clab);
mnt_csp= setElectrodeMontage(cnt.clab);

epo=makeEpochs(cnt,mkk,[0 train_ival-5]);
epo=proc_selectClasses(epo, 'not','stop');
%clear cnt cnt_lap
%epo.className

% Generate frequencies and their harmonics
freq = [];base_freq=[];

for i=1:size(epo.y,1)
  base_freq = [base_freq sscanf(epo.className{i},'%f')];
  freq = [freq sscanf(epo.className{i},'%f')];
  for k=1:NHarmonics
    freq = [freq freq(end-k+1)*(k+1)];
  end
end
freq= unique(freq(freq<(epo.fs/2)));

frequency_matrix = zeros(size(base_freq,2),size(freq,2));
for i=1:size(base_freq,2)
  if USEALLHARMONICS==0
    frequency_matrix(i,find(mod(freq/base_freq(i),1)==0,NHarmonics+1))=1;
  else
    frequency_matrix(i,:)= mod(freq/base_freq(i),1)==0;
  end
end
%sum(frequency_matrix,2)

a_array=zeros(length(freq),opt.filtOrder*2+1);
b_array=zeros(length(freq),opt.filtOrder*2+1);
bands=[];
clear a_cell b_cell

for i=1:length(freq)
  bands(i,:)=[freq(i)-opt.filter_width freq(i)+opt.filter_width];
  [dum_b,dum_a]= butter(opt.filtOrder, bands(i,:)/epo.fs*2);
  b_array(i,:)=dum_b;
  a_array(i,:)=dum_a;
  b_cell{i}=dum_b;
  a_cell{i}=dum_a;
  clear dum_b dum_a
end
%size(a_cell)
clear cnt_lap spec_lap clear fv fv0
%clear cnt
%% define a method for obtaining the (individual/global) spatial
%% filters
clear a b LOSS_LDA LOSS_RLDAshrink fv2 hlp_w la A
N.cl=size(epo.y,1);
for i=1:size(epo.y,1)
  fv{i}.x=[];
  inx=find(frequency_matrix(i,:));
  for kkk=1:length(inx)
    b{kkk}=b_cell{inx(kkk)};
    a{kkk}=a_cell{inx(kkk)};
  end
   %disp('temporal filterbank') 
  size(a);size(b);size(epo.x);
  epo_flt=proc_filterbank(epo,b,a);
  epo_flt.x=reshape(epo_flt.x,size(epo.x,1),size(epo.x,2),size(epo.x,3),size(a,2));
  if opt.CSP4EACHFREQ==0
    epo_flt.x=permute(epo_flt.x,[1 2 4 3]);
    epo_flt.x=reshape(epo_flt.x,size(epo_flt.x,1),size(epo_flt.x,2)*size(epo_flt.x,3),size(epo_flt.x,4));
    hlp_w{i}=zeros(size(epo_flt.x,2),4);
    epo_csp=epo_flt;
    epo_csp=proc_combineClasses(epo_csp,epo_csp.className{i},{epo_csp.className{setdiff(1:length(epo_csp.className),i)}});
    epo_csp.y(2,epo_csp.y(2,:)~=0)=1;
    opt_xv= strukt('loss',{'classwiseNormalized', [sum(epo_csp.y(1,:)==1); sum(epo_csp.y(1,:)==0)]}, ...
      'std_of_means',0,'verbosity',0);
    %if inside==1
    %  [loss,loss_std] = xvalidation(epo_csp, opt.model,opt_xv, 'proc',proc);
    %end
    [fv{i}, hlp_w{i}, la{i}, A{i}]= proc_csp3(epo_csp, 'patterns',opt.nPat, 'scaling','maxto1');
    %if withgraphic==1
    %  figure;
    %  plotCSPanalysis(epo_csp, mnt, hlp_w{i}, A{i}, la{i}, 'mark_patterns', opt.usedPat);
    %end
  else
    %hlp_w{i}=zeros(size(epo_flt.x,2),12);
    for jjj=1:NHarmonics+1
      epo_csp=epo_flt;
      
      epo_csp.x=epo_csp.x(:,:,:,jjj);
      epo_csp=proc_combineClasses(epo_csp,epo_csp.className{i},{epo_csp.className{setdiff(1:length(epo_csp.className),i)}});
      epo_csp.y(2,epo_csp.y(2,:)~=0)=1;
      opt_xv= strukt('loss',{'classwiseNormalized', [sum(epo_csp.y(1,:)==1); sum(epo_csp.y(1,:)==0)]}, ...
        'std_of_means',0,'verbosity',0);
      [fv0, csp, dum1, dum2]= proc_csp3(epo_csp, 'patterns',opt.nPat, 'scaling','maxto1');
      if withgraphic==1
        %figure;
        %plotCSPanalysis(epo_csp, mnt_csp, csp, dum1, dum2, 'mark_patterns', opt.usedPat);
      end
      if jjj==1
        fv{i}=fv0;
        hlp_w{i}=csp;
      else
        fv{i}.x=cat(2,fv{i}.x,fv0.x);
        hlp_w{i}=blkdiag(hlp_w{i},csp);
        %size(hlp_w{i})
      end 
    end    
  end
  
  fv2{i}=proc_variance(fv{i});
  fv2{i}=proc_logarithm(fv2{i});
  LOSS(i)=xvalidation(fv2{i},opt.model,opt_xv);
end
size(a_cell)
analyze=[];
analyze.csp_a=a_cell;
analyze.csp_b=b_cell;
analyze.freq_matrix= frequency_matrix;
analyze.csp_w=hlp_w;
analyze.features=fv2;
LOSS=100*LOSS;

fprintf('mean loss: %3.2f percent\n',mean(LOSS))
[val,inx]=sort(LOSS);
fprintf('best classes (best class first):\n') 
fprintf('%s  ',epo.className{inx})
fprintf('\n')
%end