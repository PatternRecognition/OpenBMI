% test the getCondition file.

% use this to do arithmetic calculations:
cl = cell(1,3);
[cl{:}] = deal(1,2,3);
cp = cell(1,5);
[cp{:}] = deal(12,34,56,78,90);
[flag, timeshift] = getCondition('F((14+34)*23);',[],cl,100)
[flag, timeshift] = getCondition('F(cl{1}+cl{2});',[],cl,100)
[flag, timeshift] = getCondition('F(condition_param{1}+condition_param{2});',cp,cl,100)
% do comparisons:
[flag, timeshift] = getCondition('F(100*10==1000);',[],cl,100)
[flag, timeshift] = getCondition('F(100*10<1000);',[],cl,100)
[flag, timeshift] = getCondition('F(cl(1)+cl(2)==cl(3));',[],cl,100)
% combine truth values
%'and'
[flag, timeshift] = getCondition('A(F(1);,F(0);)',[],cl,100)
[flag, timeshift] = getCondition('A(F(1);,F(1);)',[],cl,100)
[flag, timeshift] = getCondition('A(F(0);,F(0);)',[],cl,100)
[flag, timeshift] = getCondition('A(F(0);,F(1);)',[],cl,100)
%'or'
[flag, timeshift] = getCondition('O(F(1);,F(0);)',[],cl,100)
[flag, timeshift] = getCondition('O(F(1);,F(1);)',[],cl,100)
[flag, timeshift] = getCondition('O(F(0);,F(0);)',[],cl,100)
[flag, timeshift] = getCondition('O(F(0);,F(1);)',[],cl,100)
%'not'
[flag, timeshift] = getCondition('N(F(1);)',[],cl,100)
[flag, timeshift] = getCondition('N(F(0);)',[],cl,100)

% test the marker conditions:
%'
adminMarker('init',struct('log',0));
adminMarker('add',1500,-110,1);
adminMarker('add',1500,-50,2);
adminMarker('add',1500,-25,3);

ival = [-100 0];
adminMarker('query',ival);
[flag, timeshift] = getCondition('M({{1,2,3},[-60]});',[],cl,100)
[flag, timeshift] = getCondition('M({{1},[-60]});',[],cl,100)
[flag, timeshift] = getCondition('M({{''**1'',2},[-60]});',[],cl,100)
[flag, timeshift] = getCondition('M({{''**1''},[-60]});',[],cl,100)
[flag, timeshift] = getCondition('M({{''**1''},[-160]});',[],cl,100)
[flag, timeshift] = getCondition('M({{''**1''},[-160]});',[],cl,150)
[flag, timeshift] = getCondition('M({{1,2,3},[-30]});',[],cl,100)
[flag, timeshift] = getCondition('M({{1,2,3},[-10]});',[],cl,100)


% now some intervals "in the future:
[flag, timeshift] = getCondition('M({{1,2,3},[0 150]});',[],cl,150)
[flag, timeshift] = getCondition('M({{1},[10 150]});',[],cl,100)
[flag, timeshift] = getCondition('M({{''**1''},[0 150]});',[],cl,100)
[flag, timeshift] = getCondition('M({{''**1''},[0 150]});',[],cl,150)
