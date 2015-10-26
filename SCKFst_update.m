function [xkk,Skk,o1,nis] = SCKFst_update(xkk1,Skk1,W,o1,yk)

global Rs nin nhid;

%==========================================================================
% Potter's squre-root Meas. update
%==========================================================================

H = [zeros(1,nin-1) 1];

phi = Skk1'*H';

a = 1/(phi'*phi + Rs);

gam = 1/(1+(a*Rs)^(0.5));

Skk = Skk1*(eye(nin) - a*gam*phi*phi');

G = a*Skk*phi;    

resid = (yk - H*xkk1);      %residual

xkk = xkk1 + G*resid;  

nis = a*resid^2;        %normalized innov. squared

%==========================================================================
% update the internal state
%==========================================================================

W1 = reshape( W(1:nhid*(nin+1+nhid)),nhid,nin+1+nhid );
    
u1 = W1*[xkk;1;o1];

o1 = (2./(1+exp(-2*u1)) - 1);   

