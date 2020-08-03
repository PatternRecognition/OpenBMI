function stim_selfpaced(varargin)
%stim_selfpaced(<OPT>)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'filename', 'selfpaced', ...
                  'test', 0, ...
                  'bv_host', 'localhost', ...
                  'maxtime', 600, ...
                  'crossColor',0.7*[1 1 1], ...
                  'crossLineWidth',4, ...
                  'stimPos',0.55, ...
                  'stimSize',0.25, ...
                  'stimColor',[0 0 0], ...
                  'maxExtent',1, ...
                  'breakFactor',1, ...
                  'validKeys','asdfjklö');

global VP_CODE

if ~isempty(opt.bv_host),
  bvr_checkparport;
end

validKeys= opt.validKeys;

if ~opt.test & ~isempty(opt.filename),
  bvr_startrecording([opt.filename VP_CODE]);
  pause(1);
end

ppTrigger(251);

key= NaN;
ei= 0;
state= acquire_bv(1000, opt.bv_host);
figure(2);clf;
hold on;
numPressed= zeros(1,length(opt.validKeys));
for ii = 1:length(opt.validKeys)
  t(ii)=text(ii,1,opt.validKeys(ii));
  num(ii)=text(ii,0,sprintf('%i',numPressed(ii)));
end
set(t,'FontSize',15)

t_last=text(0,-1,sprintf('Last Diff: %4.0f',0));
t_win=text(0,-2,sprintf('Average Diff: %4.0f',0));

axis([0 length(opt.validKeys)+1 -3 2]);
axis off;
response=[];
t0= clock;
t1=t0;
elap=0;
while elap<opt.maxtime 
     key= NaN;
     [dmy]= acquire_bv(state);  %% clear the queue
     while ~ismember(key, validKeys) & elap<opt.maxtime,
       [dmy,bn,mp,mt,md]= acquire_bv(state);
       for mm= 1:length(mt),
         switch(mt{mm}),
           case 'R  1',
             key= 'a';
             continue;
           case 'R  2',
             key= 's';
             continue;
           case 'R  4',
             key= 'd';
             continue;
           case 'R  8',
             key= 'f';
             continue;
           case 'R 16',
             key= 'j';
             continue;
           case 'R 32',
             key= 'k';
             continue;
           case 'R 64',
             key= 'l';
             continue;
           case 'R128',
             key= 'ö';
             continue;
         end
       end
       elap = etime(clock,t0);
       pause(0.001);  %% this is to allow breaks
     end
     ei=ei+1;
     numPressed(findstr(opt.validKeys,key))= numPressed(findstr(opt.validKeys,key))+1;
     response(ei)= etime(clock,t1);
     t1= clock;
     if length(response)>0
       set(t_last,'String',sprintf('Last Diff: %4.0f',1000*response(end)));
       set(t_win,'String',sprintf('Average Diff: %4.0f',1000*mean(response)));
       for ii=1:length(t)
         set(num(ii),'String',sprintf('%i',numPressed(ii)));
       end
     end
     drawnow;
  end

   ppTrigger(254);
  bbciclose;

pause(5);
if ~opt.test & ~isempty(opt.filename),
  bvr_sendcommand('stoprecording');
end
