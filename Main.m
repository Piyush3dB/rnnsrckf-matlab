clc;
close all;
clear all;

randn('seed',unidrnd(100));

randn('state',sum(100*clock)); 

global  Qs Rs Rw nin nout nhid dE tau nwts lambda;

%=========================================================================
%  user-defined parameters
%=========================================================================

nexpt = 50;

nrun = 10;

% Number of training samples
nex_tn  = 100;
nex_tst = 100;

lambda = 1-5e-4; 

tau = 1;        % embedding delay

Lp  = 20;       % length of priming time in test

Lw  = 10;       % length of sweeping window

nin = 7;

nout = 1;

nhid = 5;

nwts = nhid*(nin+1+nhid)+nout*(nhid+1);

dE = nin;       % embedding dim.

%==========================================================================
% Generate input-output data:
%==========================================================================

% Training Data
load TN_noisy ;
load TN_clean ;

% Test Data
load tt_noisy ;
load tt_clean ;

% Signal power aka variance?
load sig ;

Rs = sig^2;

Qs = 0.1*Rs;

Rw = Rs + Qs; 

PtrArray = ceil((500-(nin+nout+nex_tn))*rand(1,nrun));

TTSTAT = [];

tic; 

for expt = 1:nexpt
    
    fprintf('=====================================\n');
    fprintf('expt in process = %d\n',expt); 
    fprintf('=====================================\n');    
   
    W1 = ( rand(nhid,nin+1+nhid) - 0.5 )*sqrt(3/(nin+1+nhid));
    W2 = ( rand(nout,nhid+1) - 0.5 )*sqrt(3/(nhid+1));
    W = [W1(:); W2(:)];
    
    wkk = W;
    
    Swkk = sqrt(5e-2*diag( [1*ones(nhid*(nin+1+nhid),1); 10*ones(nout*(nhid+1),1) ]));       
   
    for run = 1:nrun
        
        fprintf('run in process = %d\n',run); 
        
        % Get the data
        ptr = PtrArray(run);
        tn_clean = TN_clean(ptr:ptr+nex_tn-1);
        tn_noisy = TN_noisy(ptr:ptr+nex_tn-1);
    
        %==================================================================
        % Training :
        %==================================================================
        
        xkk = zeros(nin,1);
        
        Sxkk = sqrt(eye(nin));        
            
        xestArray = [];
        
        o1 = zeros(nhid,1);
   
        for ex = 1:nex_tn
            
            xk1k1 = xkk;
            
            wk1k1 = wkk;
                          
            [xkk1,Sxkk1] = SCKFst_predict(xkk,Sxkk,wkk,o1);            
        
            [wkk,Swkk] = SCKFwt(wkk,Swkk,xk1k1,o1,tn_noisy(ex));
            
            [xkk,Sxkk,o1] = SCKFst_update(xkk1,Sxkk1,wk1k1,o1,tn_noisy(ex)); 
            
            % Save Training results
            xestArray = [xestArray xkk(end)];
            
            
        end; %end of tn exampls
        
        MSE_tn(expt,run) = mse(tn_clean' - xestArray);
       
        clf;

        figure(1);
        subplot(2,1,1)
        plot(tn_clean,'k','linewidth',2);
        hold on;
        plot(tn_noisy,'r*','markersize',6);
        plot(xestArray,'k--','linewidth',2);
        hold off;

        subplot(2,1,2)
        plot(MSE_tn(end,1:run),'r-o','linewidth',2);
               
        %==================================================================
        % Testing :
        %==================================================================
        
        xkk_tt = zeros(nin,1);
    
        Sxkk_tt = sqrt(eye(nin));          
        
        xestArray_tt = [];
        
        o1_tt = zeros(nhid,1);

        for ex = 1:nex_tst % Iterate through test cases
           
            [xkk1_tt,Sxkk1_tt] = SCKFst_predict(xkk_tt,Sxkk_tt,wkk,o1_tt); 
        
            [xkk_tt,Sxkk_tt,o1_tt,nis] = SCKFst_update(xkk1_tt,Sxkk1_tt,wkk,o1_tt,tt_noisy(ex));           
            
            % Save Testing results
            xestArray_tt = [xestArray_tt xkk_tt(end)];
            
            nisArray(ex) = nis;
            
   
        end;    %end of tt exampls
        
        MSE_tt(expt,run) = mse(tt_clean' - xestArray_tt);
        
        figure(2);
        subplot(2,1,1);
        plot(tt_clean,'k','linewidth',2);
        hold on;
        plot(tt_noisy,'r*','markersize',6);
        plot(xestArray_tt,'k--','linewidth',2);
        hold off;

        figure(2);
        subplot(2,1,2);
        plot(MSE_tt(end,1:run),'r-o','linewidth',2);        
          
        drawnow;    
 
    end; % end of all epochs
    
    ttstat = [];
    
    for i = Lw+Lp:100
        
        ttstat = [ttstat  mean(nisArray(i+1-Lw:i))];
        
    end; 
    
    TTSTAT = [TTSTAT; ttstat];
    
end;    % end of expt

figure(3);
plot(mean(TTSTAT,1),'k','linewidth',2);
hold on;
DOWNTH = chi2inv(0.025,Lw*nexpt)/(Lw*nexpt);
UPTH = chi2inv(0.975,Lw*nexpt)/(Lw*nexpt);
plot(DOWNTH*ones(1,size(TTSTAT,2)),'r--','markersize',6);
plot(UPTH*ones(1,size(TTSTAT,2)),'r--','markersize',6);
%axis([41 100 0.5 20]);
hold off;

%==========================================================================
% Plot
%==========================================================================

MSE_tn_sckf = MSE_tn;
MSE_tt_sckf = MSE_tt;


xestArray_sckf = xestArray;
xestArray_tt_sckf = xestArray_tt;


TTSTAT_sckf = TTSTAT;

figure;
plot(tn_clean,'k','linewidth',2);
hold on;
plot(tn_noisy,'r*','markersize',6);
plot(xestArray_sckf,'k--','linewidth',2);
hold off;

figure;
plot(tt_clean,'k','linewidth',2);
hold on;
plot(tt_noisy,'r*','markersize',6);
plot(xestArray_tt_sckf,'r--','linewidth',2);

figure;
plot(mean(MSE_tn_sckf,1),'r-o','linewidth',2);

figure;
plot(mean(MSE_tt_sckf,1),'r-o','linewidth',2);

figure;
plot(DOWNTH*ones(1,size(TTSTAT_sckf,2)),'k--','markersize',6);
hold on;
plot(UPTH*ones(1,size(TTSTAT_sckf,2)),'k--','markersize',6);
plot(mean(TTSTAT_sckf,1),'r','linewidth',2);
legend('Lower Conf. limit','Upper Conf. Limit','NIS')
