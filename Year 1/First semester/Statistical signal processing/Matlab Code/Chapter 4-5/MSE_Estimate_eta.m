%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab calcola la stima MM ed ML del parametro Eta 
% della ddp di Laplace, ne calcola l'MSE e lo confronta con il CRB
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Nvect=[2:2:1000]; % Sample size
Nd=length(Nvect);
M=10^4;           % Numero di prove Monte Carlo dell'esperimento

A=2;              % valor medio v.a. Laplace
Px=1;           % potenza rumore di Laplace
eta=sqrt(Px/2);   % parametro della v.a. di Laplace per ottenere Px

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 h=waitbar(0,'Please wait...');
 
for i=1:Nd
    
    waitbar(i/Nd,h); 
    N=Nvect(i);
    U=rand(N,M);
    XE=-eta*log(U); 
    S=sign(rand(N,M)-0.5);
    XL=S.*XE+A;
    vmL=mean(XL);
    medL=median(XL);
    PxMM=mean((XL-vmL).^2);
    etahatMM=sqrt(PxMM./2);
    etahatML=mean(abs(XL-medL));
    MSE_Eta_MM(i)=sum((etahatMM-eta).^2)/M;
    MSE_Eta_ML(i)=sum((etahatML-eta).^2)/M;

end

close(h)


CRB_L=(eta^2)./Nvect;

figure (1);
loglog(Nvect,CRB_L,'green', 'LineWidth',2);grid on;
hold on;
loglog(Nvect,MSE_Eta_MM,'blue', 'LineWidth',2);
loglog(Nvect,MSE_Eta_ML,'red', 'LineWidth',2);
legend('CRB(\eta)', 'MSE MM Estimate', 'MSE ML Estimate');
title('Estimate of \eta - Laplace noise');xlabel('Sample size (N)');ylabel('Mean Square Error (MSE)')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Autore: Fulvio Gini 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: fulvio.gini@unipi.it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 