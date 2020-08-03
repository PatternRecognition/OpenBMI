function mrk_paced= getPacedAltEventsNew(Mrk, mrk_array, clickSide, tol_msec)
%mrk_paced= getPacedAltEventsNew(mrk, mrk_array, clickSide, <tol>)
%
% getPacedAltEvents: get non-error markers of a paced_alt experiment
%
% IN  mrk       - 
%     mrk_array - [tick_trg, tock_trg, <click_trg>]
%                 default for click_trg is -'F' for clickSide='left,
%                 and -'J' otherwise
%     clickSide - a string, 'left' or 'right'
%     tol       - in msec (click must be within this range around tick),
%                 default [250 700]

if ~exist('tol_msec','var'), tol_msec=[250 700]; end
tol= tol_msec/1000*Mrk.fs;

tick= mrk_array(1);
tock= mrk_array(2);
if length(mrk_array)>2,
  click_trg= mrk_array(3);
elseif strcmp(clickSize, 'left'),
  click_trg= -'F';
else
  click_trg= -'J';
end

classDef= {tick,tock,click_trg; 'tick','tock',[clickSide ' click']};
mrk= makeClassMarkers(Mrk, classDef, 0, 0);
[dum,ev_stm]= mrk_selectClasses(mrk, 'tick');
[dum,ev_nil]= mrk_selectClasses(mrk, 'tock');
[dum,ev_rsp]= mrk_selectClasses(mrk, '* click');
po_stm= mrk.pos(ev_stm);
po_nil= mrk.pos(ev_nil);
po_rsp= mrk.pos(ev_rsp);
po_jit= repmat(po_rsp,[length(po_stm),1])-repmat(po_stm',[1,length(po_rsp)]);
pabs= abs(po_jit);
[dd,ind]= min(pabs, [], 2);
valid_click= find(dd<tol(1));
valid_noclick= find(dd>tol(2));
mrk.latency= repmat(NaN, size(mrk.pos));
for kk= 1:length(valid_click),
  mrk.latency(ev_stm(valid_click(kk)))= po_jit(valid_click(kk), ind(valid_click(kk)));
end
mrk.indexedByEpochs= {'latency'};

po_jit= repmat(po_rsp,[length(po_nil),1])-repmat(po_nil',[1,length(po_rsp)]);
pabs= abs(po_jit);
[dd,ind]= min(pabs, [], 2);
valid_rest= find(dd>tol(2));

mrk_paced= mrk;
mrk_paced.toe(ev_stm(valid_click))= -click_trg;
state= bbci_warning('off', 'mrk');
mrk_paced= mrk_selectEvents(mrk_paced, ...
                            [ev_stm([valid_click; valid_noclick]) ...
                             ev_nil(valid_rest)], 'sort',1);
bbci_warning(state);


if strcmpi(clickSide, 'left'),
  classDef = {-click_trg,tick,tock; 'left click','right no-click','rest'};
else
  classDef = {tick,-click_trg,tock; 'left no-click','right click','rest'};
end
mrk_paced= makeClassMarkers(mrk_paced, classDef, 0, 0);
