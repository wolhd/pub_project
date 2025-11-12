f = [ -2 -3 -4 -5 ];
intcon = [1 1 1];
Aeq = [ 1 1 1 0;
        1 0 0 1;
        0 1 1 0;];

beq = [ 1 1 1];
 
[x,fval,exitflag,output] =  intlinprog(f,intcon,[],[],Aeq,beq,[0 0 0 0],[1 1 1 1]);
