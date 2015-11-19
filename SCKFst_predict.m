function [xkk1,Skk1] = SCKFst_predict(xkk,Skk,W,o1)

global nhid nin nout Qs lambda;

nx = nin;

nPts = 2*nx;

QPtArray = sqrt(nx)*[eye(nx)  -eye(nx)];

Xi = repmat(xkk,1,nPts) + Skk*QPtArray;

Xi = StateEq(Xi,W,o1);

xkk1 = sum(Xi,2)/nPts; 

X = (Xi-repmat(xkk1,1,nPts))/sqrt(nPts);

Qsqrt = sqrt( diag([zeros(1,nin-1) Qs]) );

[foo,Skk1] = qr([ X Qsqrt]',0);

Skk1 = Skk1'; 

%=========================================================================

function Xout = StateEq(X,W,o1)

global nhid nin nout;

W1 = reshape( W(1:nhid*(nin+1+nhid))    ,  nhid, nin+1+nhid );
    
W2 = reshape( W(nhid*(nin+1+nhid)+1:end),  nout, nhid+1     );

nPts = 2*nin;

v2 = ones(1,nPts);
v3 = repmat(o1,1,nPts);

Ar = [ X ; v2 ; v3 ];

u1 = W1 * Ar;

o1_n = (2./(1+exp(-2*u1)) - 1);  
    
dhat = W2*[o1_n;ones(1,nPts)];
    
Xout = [X(2:end,:); dhat];   
    

